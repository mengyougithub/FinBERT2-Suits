
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import cast
from dataclasses import dataclass, fields
import random
import pandas as pd

def load_ner_data(file_path):
    texts_list = []
    labels_list = []
    texts = []
    labels= []
    records_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line=="\n":
                texts_list.append(texts)
                labels_list.append(labels)
                texts = []
                labels= []
            else:
                parts = line.strip().split()
                text = parts[0] 
                label = parts[1] 
                texts.append(text) # 用空格连接文本
                labels.append(label)  # 用空格连接标签
        i=0
        for tokens,nertags in zip(texts_list, labels_list):
            nerdic={'id': i,'tokens':tokens,'ner_tags':nertags}
            records_list.append(nerdic)
            i+=1
    return records_list

# train_texts, train_labels = load_ner_data(file_path1)
# file_path2 = '/root/autodl-tmp/xuxuan/downstream_data/test_company.txt'
# test_texts, test_labels = load_ner_data(file_path2)

class companyNERDataset(Dataset):
    def __init__(self, file_path1):
        self.records = load_ner_data(file_path1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
    def getlabel2id(self):
            # 使用LabelEncoder来获取每个标签的整数ID
            labels=[record["ner_tags"] for record in self.records]
            alllabels=[]
            for label in labels:
                alllabels.extend(label)
            label_encoder = LabelEncoder()
            # labels = [record.label for record in self.records]

            int_labels = label_encoder.fit_transform(alllabels)
            # 创建ID到标签的映射
            id2label = {int(id_): label for label, id_ in zip(alllabels, int_labels)}
            label2id = {label: int(id_) for label, id_ in zip(alllabels, int_labels)}

            # print(id2label)
            return label2id , id2label 


def save_dataset_to_csv(dataset, output_filename):
    """
    将给定的Dataset对象保存为CSV文件。

    参数:
    - dataset: 一个继承自torch.utils.data.Dataset的Dataset对象。
    - output_filename: 保存CSV文件的文件名。
    """
    # 将Dataset转换为列表
    data_list = [item for item in dataset]
    
    # 将列表转换为DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存DataFrame到CSV文件
    df.to_csv(output_filename, index=False)
    print(f"Dataset has been saved to {output_filename}")
