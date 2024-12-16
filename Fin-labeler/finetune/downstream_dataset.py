
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import cast
from dataclasses import dataclass, fields
import random
import pandas as pd

@dataclass(slots=True)
class PairRecord:
    text: str
    label: str

def load_FinSentimentData(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除行尾的换行符并分割文本和标签
            parts = line.strip().split('    ')
            # 过滤掉空字符串
            parts = [part for part in parts if part]

            # 提取文本和标签
            text = parts[0]  
            label=parts[1]  # 转换为PyTorch张量

            record=PairRecord(text=text,label=label)
            records.append((record))  
    return records


class SentimentDataset(Dataset):
    def __init__(self, file_path, shuffle=True):
        self.records = self._load_data(file_path)
        if shuffle:
            # 打乱数据
            random.shuffle(self.records)
    def _load_data(self, file_path):
        """
        加载数据文件并解析每一行。
        """

        records = load_FinSentimentData(file_path)
        return records

    def __len__(self):
        """返回数据集的大小"""
        return len(self.records)

    def __getitem__(self, idx):
        """获取单个样本"""
        # text,label =  self.record_list[idx].text, self.record_list[idx].label
        record=  self.records[idx]
        # return text,label
        return record
    def getlabel2id(self):
        # 使用LabelEncoder来获取每个标签的整数ID
        label_encoder = LabelEncoder()
        labels = [record.label for record in self.records]

        int_labels = label_encoder.fit_transform(labels)
        # 创建ID到标签的映射
        id2label = {int(id_): label for label, id_ in zip(labels, int_labels)}
        label2id = {label: int(id_) for label, id_ in zip(labels, int_labels)}

        print(id2label)
        return label2id , id2label 

class SentimentDataset2(Dataset):
    def __init__(self, csvpath, text_column='CONCAT_TIT_ABS', label_column='SENTIMENT_FLAG'):
        self.df = pd.read_csv(csvpath)
        self.texts = self.df[text_column].tolist()
        self.labels = self.df[label_column].astype(int).tolist() # 确保标签为整数类型
        self.records = [PairRecord(text, label) for text, label in zip(self.texts, self.labels)]
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
    
    def getlabel2id(self):
        id2label = {0:"负面",1:"正面"}
        label2id = {"负面":0,"正面":1}
        print(id2label)
        return label2id , id2label 
# dataset=SentimentDataset2("/root/autodl-tmp/xuxuan/downstream_data/financialnews_sentiment_test.csv")
class sentiment2_collator:
    def __init__(self, tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records):
        texts = [record.text for record in records]
        # labels = [record.label for record in records]
        labels = [int(record.label) for record in records]

        inputs = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_ids = inputs['input_ids']
        
        labels =torch.tensor(labels, dtype=torch.long)
        
        # text_ids = cast(torch.Tensor, text_ids)
        # labels = cast(torch.Tensor, labels)
       
        return {
            'input_ids': text_ids,
            'labels': labels
        }
def load_industry_data(file_path):
    # text_list = []
    # label_list = []
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除行尾的换行符并分割文本和标签
            parts = line.strip().split('	')
            # 过滤掉空字符串
            parts = [part for part in parts if part]
            # 提取文本和标签
            text = parts[1]  
            label = parts[0]  
            record=PairRecord(text=text,label=label)
            records.append((record))  
    return records

class industryDataset(Dataset):
    def __init__(self, file_path, shuffle=True):
        self.records = self._load_data(file_path)
        if shuffle:
            random.shuffle(self.records)
    def _load_data(self, file_path):
        """
        加载数据文件并解析每一行。
        """
        records = load_industry_data(file_path)

        return records

    def __len__(self):
        """返回数据集的大小"""
        return len(self.records)

    def __getitem__(self, idx):
        """获取单个样本"""
        # text,label =  self.record_list[idx].text, self.record_list[idx].label
        record=  self.records[idx]
        # return text,label
        return record
    def getlabel2id(self):
        # 使用LabelEncoder来获取每个标签的整数ID
        label_encoder = LabelEncoder()
        labels = [record.label for record in self.records]

        int_labels = label_encoder.fit_transform(labels)
        
        # 创建ID到标签的映射
        id2label = {id_: label for label, id_ in zip(labels, int_labels)}
        label2id = {label: id_ for label, id_ in zip(labels, int_labels)}

        return label2id , id2label


class NERDataset(Dataset):
    def __init__(self, file_path):
        self.texts,self.labels = self._load_data(file_path)

    def _load_data(self, file_path):
        """
        加载数据文件并解析每一行。
        """
        texts, labels = load_industry_data(file_path)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = np.array(labels)

        return texts,labels

    def __len__(self):
        """返回数据集的大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """获取单个样本"""
        text,label =  self.texts[idx],self.labels[idx]
        return text,label




class sentiment_collator:
    def __init__(self, tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records):
        texts = [record.text for record in records]
        labels = [int(record.label) for record in records]

        inputs = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_ids = inputs['input_ids']
        
        labels =torch.tensor(labels, dtype=torch.long)
       
        return {
            'input_ids': text_ids,
            'labels': labels
        }
        
class industry_collator:
    def __init__(self, tokenizer, max_length,label2id) :
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length
        self.label2id=label2id
    def __call__(self, records):
        texts = [record.text for record in records]
        labels = [record.label for record in records]
        inputs = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_ids = inputs['input_ids']
        labels= [self.label2id[label] for label in labels]
        labels =torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': text_ids,
            'labels': labels
        }
