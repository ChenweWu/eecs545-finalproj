import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from mgca.constants import *
from mgca.datasets.utils import get_imgs, get_imgs_single
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import CLIPTokenizer
import pandas as pd
import json

def read_json_files(folder_dir):
    data_frame = {}
    
    folder_path = folder_dir + "/text"
    for idx, file in enumerate(os.listdir(folder_path)):
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            print(f"current idx of file: {idx}")
            with open(file_path, 'r', encoding='utf-8') as json_file:
                
                data = json.load(json_file)
                if data_frame == {}:
                    for key in data.keys():
                        data_frame[key] = [data[key]]
                else:
                    for key in data.keys():
                        data_frame[key].append(data[key])
    
    with open(folder_dir + "/filenames.json", 'w') as json_file:
        json.dump(data_frame, json_file)
    
class SynPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=512, max_words=77, sent_num=3, is_single = 0):
        super().__init__()
        self.is_single = is_single
        filename = "syn_ours_direct_embed_short"
       # 2024-04-29T17-22-27_ours /

        file_path = f"/scratch/bcde/ztshuai/{filename}/filenames.json"
        
        self.file_path = file_path
        if not os.path.exists(file_path):
            path = f"/scratch/bcde/ztshuai/{filename}"
            print(f"WARNING: This is {path}!!!!!!!!!!!!!!!!!!!")
            read_json_files(path)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        self.filenames = pd.DataFrame(data)
        print(self.filenames.loc[0]['path'])
        print(f"syn dataset length {len(self.filenames)}")
        self.max_words = max_words
        self.imsize = imsize
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained("/scratch/bcde/ztshuai/roentgen/tokenizer")
    def __len__(self):
        return len(self.filenames)

    def get_caption(self, sent):
        
        
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        '''
        jsonfile = self.filenames.iloc[index]
        key = jsonfile['path']
        sent = jsonfile['report']
        meta_data = jsonfile['pa']
        '''
        
        key = self.filenames['path'][index]
        key = key.replace("roentgen","roentgen_short")
        sent = self.filenames['report'][index]
        meta_data = torch.tensor(self.filenames['pa'][index])
        caps, cap_len = self.get_caption(sent)
        if self.is_single == 0:
            imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        else:
            imgs = get_imgs_single(key, self.imsize, self.transform, multiscale=False)
        
        
        return imgs, caps, cap_len, key, meta_data.float()

    
def syn_multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention, meta_datas = [], [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p, meta_data = b
        #print(img.shape) # 3*512*512
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        #tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        meta_datas.append(meta_data)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    #tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    meta_datas = torch.stack(meta_datas)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)
    
    path = np.array(path)
    #print("meta_datas:", meta_datas.shape)
    return_dict = {
        "caption": ids[sorted_cap_indices],
        #"token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "image": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
        "meta_data": meta_datas[sorted_cap_indices]
    }
    return return_dict