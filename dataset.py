import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    '''
    繼承torch.Dataset
    '''
    def __init__(self,seq_paris, tokenizer , special_tokens_dict , pad_idx , mode = 'train'):
        self.seq_paris = seq_paris
        self.tokenizer = tokenizer
        self.special_tokens_dict = special_tokens_dict
        self.pad_idx = pad_idx
        self.mode = mode

    def __len__(self):
        return len(self.seq_paris)

    def __getitem__(self, index):
        return self.seq_paris[index]

    def pad_sequence(self , non_pad_token , non_pad_label , non_pad_attn):
        '''
        input : token ids, labels, attention masks
        將每個向量 padding 之後組成矩陣
        output : pad token ids, pad labels, pad attention masks
        '''
        max_size = max([len(ele) for ele in non_pad_token])
        pad_batch1 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_token])
        pad_batch2 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_label])
        pad_batch3 = torch.stack([torch.cat([t, torch.LongTensor([0] * (max_size - len(t)))]) for t in non_pad_attn])
        return pad_batch1 , pad_batch2 , pad_batch3

    def collate_batch(self , datasets):
        '''
        input : token ids
        Dataloader 呼叫時的函式
        回傳批次的torch data
        output : pad token ids, pad labels, pad attention masks
        '''
        tokens_list , labels_list , attention_mask_list = [] , [] , []
        for dataset in datasets:
            self.tokenizer.padding_side = 'left'
            encoded_seq = self.tokenizer(dataset, padding=True)
            indexed_tks = encoded_seq["input_ids"]
            attention_mask = encoded_seq["attention_mask"]

            tokens_list.append(torch.tensor(indexed_tks))
            labels_list.append(torch.tensor(indexed_tks))
            attention_mask_list.append(torch.tensor(attention_mask))
        return self.pad_sequence(tokens_list , labels_list , attention_mask_list)