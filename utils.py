import os, random
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
from torch.nn import functional as F

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_torch_seed()

def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案

    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)

    bounary , item_idx , temp_seq , seq_pairs = 0 , 0 , "" , []

    for w_idx, word in enumerate(article):
        if word == '\n':
            new_line_idx = w_idx + 1
            if temp_seq == "":
                temp_seq = "PHI:Null"

            seq_pair = special_tokens_dict['bos_token'] + article[bounary:new_line_idx] + special_tokens_dict['sep_token'] + temp_seq + special_tokens_dict['eos_token']
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\n"
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    return seq_pairs

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder, num_processes=4):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())

    pool = multiprocessing.Pool(num_processes)
    print("processing each medical file")
    results = pool.starmap(process_medical_report, [(txt_name, medical_report_folder, annos_dict, special_tokens_dict) for txt_name in txt_names])

    seq_pairs = [pair for result in results for pair in result]

    pool.close()
    pool.join()
    print("All medical file done")
    return seq_pairs

def sample_text(model, tokenizer, text, n_words=100, device="cpu"):
    '''
    input : model, tokenizer, text(句子 string), n_words(生成字數限制)
    output : 模型預測結果 (string)
    '''
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None

    with torch.no_grad():
        for _ in range(n_words):
            out = model(inputs, past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            text.append(inputs.item())
            if tokenizer.decode(inputs.item()) == eos:
                break

    return tokenizer.decode(text)

def get_anno_format(sentence , infos , boundary):
    '''
    input : 句子(string), 模型預測phi資訊(string), 上個句子結尾索引(integer)
    將模型輸出的 phi 資訊對應 sentence
    儲存於字典並append於list
    output : 儲存phi字典的list
    '''
    anno_list = []
    lines = infos.split("\n")
    # 創建一個字典，用來存儲PHI信息的對應
    normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
    phi_dict = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            phi_dict[parts[0]] = parts[1]
    for phi_key, phi_value in phi_dict.items():
        normalize_time = None
        if phi_key in normalize_keys:
            if '=>' in phi_value:
                temp_phi_values = phi_value.split('=>')
                phi_value = temp_phi_values[0]
                normalize_time = temp_phi_values[-1]
            else:
                normalize_time = phi_value
        if phi_value not in sentence or len(phi_value) < 1:
            continue
        st_idx = sentence.find(phi_value)
        ed_idx = st_idx + len(phi_value)
        item_dict = {
                    'phi' : phi_key,
                    'st_idx' : st_idx + boundary,
                    'ed_idx' : ed_idx + boundary,
                    'entity' : phi_value,
        }
        if normalize_time is not None:
            item_dict['normalize_time'] = normalize_time
        anno_list.append(item_dict)
    return anno_list

def predict_sent(sents, model=None, tokenizer=None):
    '''
    input : 一篇病理報告全部的句子(list)
    output : 上傳格式的 phi 資訊
    '''
    boundary = 0
    annotations = []
    device = model.device

    for sent in sents:
        decode_phase = sample_text(model, tokenizer, text=special_tokens_dict['bos_token'] + sent , n_words=200, device=device)
        if special_tokens_dict['sep_token'] in decode_phase:
            try:
                _ , phi_infos = decode_phase.split(special_tokens_dict['sep_token'])
            except:
                continue

            if "PHI:Null" not in phi_infos:
                annotation = get_anno_format(sent , phi_infos , boundary)
                annotations.extend(annotation)
        boundary += len(sent)
    return annotations

def predict_file(txts , write_file, model=None, tokenizer=None):
    with open(write_file , 'w' , encoding='utf-8') as fw:
        for txt in tqdm((txts)):
            test_sents = read_file(txt)
            anno_infos = predict_sent(test_sents, model=model, tokenizer=tokenizer)
            txt_name = txt.split('\\')[-1].replace('.txt' , '')
            for anno_info in anno_infos:
                fw.write(txt_name + '\t')
                fw.write(f"{anno_info['phi']}\t")
                fw.write(f"{anno_info['st_idx']}\t")
                fw.write(f"{anno_info['ed_idx']}\t")
                if anno_info['phi'] in ['DATE' , "TIME" , "DURATION" , "SET"]:
                    fw.write(f"{anno_info['entity']}\t")
                    if anno_info["normalize_time"] != "":
                        fw.write(f"{anno_info['normalize_time']}\n")
                    else:
                        fw.write(f"{anno_info['entity']}\n")
                else:
                    if len(anno_info['entity']) > 0:
                        fw.write(f"null\n")
                    else: fw.write(f"{anno_info['entity']}\n")