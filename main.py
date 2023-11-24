import os

import numpy as np
from tqdm import tqdm, trange
from torch.optim import AdamW

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    get_linear_schedule_with_warmup, 
)

import random
import matplotlib.pyplot as plt
from torch.nn import functional as F

import argparse
import wandb
from dataset import *
from utils import *

parser = argparse.ArgumentParser(description="ai cup contest")
parser.add_argument("--experiment", default="", type=str)
parser.add_argument("--tokenizer_name", default="EleutherAI/pythia-70m", type=str)
parser.add_argument("--model_name", default="EleutherAI/pythia-70m", type=str)
parser.add_argument("--model_ckpt_path", default="results/base/GPT_best.pt", type=str)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--output_dir", default="./results", type=str)
parser.add_argument("--log_to_file", default=1, type=int)
parser.add_argument("--n_epoch", default=1, type=int)
parser.add_argument("--do_train", default=0, type=int)
parser.add_argument("--do_test", default=1, type=int)

args = parser.parse_args()

data_root = "./data/First_Phase_Dataset"
anno_info_path = f"{data_root}/answer.txt"
report_folder = f"{data_root}/First_Phase_Text_Dataset"
train_seq_pairs = generate_annotated_medical_report_parallel(anno_info_path, report_folder, num_processes=4)

idx = 10

print(f"input : \n{train_seq_pairs[idx]}")

BATCH_SIZE = 12

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

tokenizer.add_special_tokens(special_tokens_dict)
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tr_dataset = GPTDataset(train_seq_pairs,
                        tokenizer,
                        special_tokens_dict,
                        PAD_IDX)

bucket_train_dataloader = DataLoader(tr_dataset,
                                    batch_size=BATCH_SIZE,
                                    collate_fn=tr_dataset.collate_batch)

model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer))
if len(args.model_ckpt_path) > 0:
    print(f"[INFO]: load model from {args.model_ckpt_path}")
    model.load_state_dict(torch.load(args.model_ckpt_path))


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=args.lr,   
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_name = args.experiment if len(args.experiment) > 0 else "base"
model_dir = os.path.join(args.output_dir, exp_name)
log_file = os.path.join(model_dir, "log.txt")

if not os.path.isdir(model_dir):
    os.makedirs(model_dir, exist_ok=True)
log_file = open(log_file, "w")
p_kwargs = dict(file=log_file, flush=True) if args.log_to_file else {}

model = model.to(device)
min_loss = 9999

predict_text = special_tokens_dict['bos_token'] + "MANILDRA  NSW  2865"

# train
if args.do_train:
    for _ in range(args.n_epoch):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for step, (seqs, labels, masks) in enumerate(tqdm(bucket_train_dataloader)):
            seqs = seqs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            model.zero_grad()
            outputs = model(seqs, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(bucket_train_dataloader)

        print("Average train loss: {}".format(avg_train_loss), **p_kwargs)
        print(sample_text(model, tokenizer, text=predict_text, device=device), **p_kwargs)
        torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_Finial.pt'))
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_best.pt'))

# predict
if args.do_test:
    model.eval()
    test_phase_path = f'{data_root}/Validation_Release'
    test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))

    write_file = os.path.join(model_dir, "answer.txt")
    print(test_txts, **p_kwargs)
    predict_file(test_txts , write_file, model=model, tokenizer=tokenizer)