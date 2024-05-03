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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class MultimodalPretrainingDataset_balance(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=512, max_words=77, sent_num=3, is_single = 0):
        super().__init__()
        self.is_single = is_single
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        #self.df = pd.read_csv(str(MIMIC_CXR_DATA_DIR)+ "/total" + "_master.csv")
        self.df = pd.read_csv(MIMIC_CXR_MASTER_RACE_CSV)
        print("columns!!!!!!!!!!!!!!!",self.df.columns)
        print(self.df['Path'])
        
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))
        self.demo_columns = ['sex_label', 'age', 'race_label']
        self.disease_columns = ['ED','PE', 'CO', 'PX','PN']
        
        
        # load studies and study to text mapping
        
        if not split == "all":
            self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("/scratch/bcde/ztshuai/roentgen/tokenizer")
        #self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        print("dataset length before balancing:", len(self.df))
        if split == "train" or split == "all":
            condition1 = self.df['ED']==1
            df1 = self.df[condition1].sample(n=2000, random_state=42)
            
            
            condition2 = self.df['PE']==1
            df2 = self.df[condition2].sample(n=2000, random_state=42)
            condition3 = self.df['CO']==1
            df3 = self.df[condition3].sample(n=2000, random_state=42)
            condition4 = self.df['PX']==1
            df4 = self.df[condition4].sample(n=2000, random_state=42)
            condition5 = self.df['PN']==1
            df5 = self.df[condition5].sample(n=2000, random_state=42)
            
            
            condition6 = self.df['ED']+ self.df['PE']+self.df['CO']+self.df['PX']+self.df['PN']==0
            df6 = self.df[condition6].sample(n=2000, random_state=42)
            self.df = pd.concat([df1,df2,df3,df4,df5,df6], axis = 0)
        self.df.reset_index(drop=True, inplace=True)
        self.filenames, self.path2sent = self.load_text_data(split)
        
        print("final dataset length: ", len(self.df))
        
    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        metadatas = []
        for row in self.df.itertuples():
            
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            
            if cur_split == split and path in path2sent:
                
                filenames.append(path)
                meta = []
                items = self.disease_columns+['sex_label','age','race_label']
                for item in items:
                    
                    meta.append(getattr(row, item))
                metadatas.append(meta)
            elif split == "all"and path in path2sent:
                filenames.append(path)
                meta = []
                items = [self.disease_columns]+['sex_label','age','race_label']
                for item in items:
                    meta.append(getattr(row, item))
                metadatas.append(meta)
        self.metadatas = metadatas
        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)
        replace_item = self.disease_columns + ['edema','pleural effusion', 'consolidation', 'pneumothorax','pneumonia'] + ['ED','PE', 'CO', 'PX','PN']
        for term in replace_item:
            if term in sent:
                sent = sent.replace(term, "_")
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
        
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        if self.is_single == 0:
            imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        else:
            imgs = get_imgs_single(key, self.imsize, self.transform, multiscale=False)
        
        ##############################################
        
        meta_data = torch.tensor(self.metadatas[index])
        
        return imgs, caps, cap_len, key, meta_data.squeeze(0).float()



class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=512, max_words=77, sent_num=3, is_single = 0):
        super().__init__()
        self.is_single = is_single
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        #self.df = pd.read_csv(str(MIMIC_CXR_DATA_DIR)+ "/total" + "_master.csv")
        self.df = pd.read_csv(MIMIC_CXR_MASTER_RACE_CSV)
        print("columns!!!!!!!!!!!!!!!",self.df.columns)
        print(self.df['Path'])
        
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))
        self.demo_columns = ['sex_label', 'age', 'race_label']
        self.disease_columns = ['ED','PE', 'CO', 'PX','PN']
        
        
        # load studies and study to text mapping
        
        if not split == "all":
            self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("/scratch/bcde/ztshuai/roentgen/tokenizer")
        #self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        self.df.reset_index(drop=True, inplace=True)
        self.filenames, self.path2sent = self.load_text_data(split)
        
        print("final length of the dataset: ",len(self.df))
        
    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        metadatas = []
        for row in self.df.itertuples():
            
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            
            if cur_split == split and path in path2sent:
                
                filenames.append(path)
                meta = []
                items = self.disease_columns+['sex_label','age','race_label']
                for item in items:
                    
                    meta.append(getattr(row, item))
                metadatas.append(meta)
            elif split == "all"and path in path2sent:
                filenames.append(path)
                meta = []
                items = [self.disease_columns]+['sex_label','age','race_label']
                for item in items:
                    meta.append(getattr(row, item))
                metadatas.append(meta)
        self.metadatas = metadatas
        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)
        replace_item = self.disease_columns + ['edema','pleural effusion', 'consolidation', 'pneumothorax','pneumonia'] + ['ED','PE', 'CO', 'PX','PN']
        for term in replace_item:
            if term in sent:
                sent = sent.replace(term, "_")
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
        
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        if self.is_single == 0:
            imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        else:
            imgs = get_imgs_single(key, self.imsize, self.transform, multiscale=False)
        
        ##############################################
        
        meta_data = torch.tensor(self.metadatas[index])
        
        return imgs, caps, cap_len, key, meta_data.squeeze(0).float()

def multimodal_collate_fn(batch):
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


if __name__ == "__main__":
    from mgca.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
