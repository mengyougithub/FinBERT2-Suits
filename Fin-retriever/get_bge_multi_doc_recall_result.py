import pymysql
import ast
import json
import random
import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from rouge import Rouge
from numpy import dot
from numpy.linalg import norm
from FlagEmbedding import FlagReranker
import pandas as pd
import re
from sklearn.metrics import precision_recall_curve
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


import numpy as np

rouge = Rouge()
model_path = './roberta_all_0724_mlm_optimized_1-5epoch_checkpoint-60441_encoder_model_test_400_600_5w_rm_dup_hn_20_range_0_200_eval_pos_neg_select_4w_add_general_shuf_warmup_ratio_01/checkpoint-1000/'

model = SentenceTransformer(model_path)
headers = {
    "Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiYmQ4Yjc2ZmUtODc0Ny00YTQ3LTlkZDQtZGViMzQ2MjQ2YzI4IiwiZXhwIjo0OTE3NzI4ODE3fQ.fU6ew1pJoi5WvpZQzkAvGGRvHfs6Kk-ZdrrB6pstGgY",
    "Content-Type": "application/json"
}

import time


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def get_embedding(content_list):
    res = model.encode(content_list)
    return res

def get_label(dic):
    labels = []
    for query,page_contents in dic.items():
        label = []
        for page_content in page_contents[2:]:
            label.append(page_content[1])
        labels.append(label.copy())
    return labels


def rank_by_cos_max_seg_sim(dic):
    shift = 380
    count = 0
    ##################### computing_content_embedding
    for question,page_contents in dic.items():
        count += 1
        if count % 10 == 0:
            print('rank_by_cos_max_seg_sim_done_with ',str(count))
        query = page_contents[0]
        rationale = page_contents[1]
        for i,page_content in enumerate(page_contents[2:]):
            max_cos_sim0 = []
            text_list = []
            for i in range(1+(len(page_content[0]))//shift):
                ele = page_content[0][(0 + i*shift):(400 + i*shift)]
                if len(ele) == 0:
                    continue
                text_list.append(ele)
            embeddings = get_embedding(text_list)
            page_content.append(embeddings)
        dic[question] = page_contents

    sorted_ix_list = []
    score_list = []
    labels = []

    ##################### computing_query_doc_similarity
    for question,page_contents in dic.items():
        question_embedding = get_embedding([question])
        max_seg_cos_sim = []
        label = []  
        
        for i,page_content in enumerate(page_contents[2:]):
            label.append(page_content[1])
            content_embeddings = page_content[2]
            cos_sim_list = []
            for content_embedding in content_embeddings:
                cos_sim = dot(question_embedding, content_embedding)[0]
                cos_sim_list.append(cos_sim)
            max_seg_cos_sim.append(max(cos_sim_list))
        for question1,page_contents in dic.items():
            if question == question1:
                continue
            label_len = len(label)
            for j,page_content in enumerate(page_contents[2:]):
                label.append(0)
                content_embeddings = page_content[2]
                cos_sim_list = []
                for ele in content_embeddings:
                    cos_sim = dot(question_embedding, ele)[0]
                    cos_sim_list.append(cos_sim)
                max_seg_cos_sim.append(max(cos_sim_list))
        labels.append(label.copy())
        sorted_ix = np.argsort(-np.array(max_seg_cos_sim))
        sorted_ix_list.append(sorted_ix)
        score_list.append(max_seg_cos_sim[:])
    return sorted_ix_list,labels,score_list



def calculate_metrics_for_thresholds(labels, scores, thresholds):
    # 初始化结果列表
    precision_results = []
    recall_results = []
    accuracy_results = []
    f1_results = []
    
    # 遍历每个阈值
    for threshold in thresholds:
        # 将分数转换为预测标签
        predicted_labels = (scores >= threshold).astype(int)
        
        # 计算精确率、召回率、准确率和F1分数
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        accuracy = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)
        
        # 将结果添加到列表中
        precision_results.append(precision)
        recall_results.append(recall)
        accuracy_results.append(accuracy)
        f1_results.append(f1)
    
    return precision_results, recall_results, accuracy_results, f1_results


def cal_roc(score_list,labels):
    thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.999]
    #specific_thresholds = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98]

    score_list = np.concatenate(score_list).tolist()
    label = np.concatenate(labels).tolist()

    precision_results, recall_results, accuracy_results, f1_results = calculate_metrics_for_thresholds(np.array(label), np.array(score_list),thresholds)
    for threshold in thresholds:
        print(f"Threshold: {threshold} ",f" Precision: {precision_results[thresholds.index(threshold)]:.2f}",f" Recall: {recall_results[thresholds.index(threshold)]:.2f}",f" Accuracy: {accuracy_results[thresholds.index(threshold)]:.2f}",f" F1 Score: {f1_results[thresholds.index(threshold)]:.2f}")
    fpr, tpr, thresholds = roc_curve(label, score_list)
    roc_auc = auc(fpr, tpr)
    print('roc_auc',roc_auc)


#计算topk召回正样本召回率，label=1为通过rouge判断的正样本，0为负样本，2不确定
def cal_rank_res(top_k,sorted_ix_list,labels,score_list):
    pos_recalled_ratio_list = []
    med_recalled_ratio_list = []

    not_recall_list = []
    wrong_recall_list = []

    not_recall_score_list = []
    wrong_recall_score_list = []

    pos_cnt_list = []
    cnt_list = []
    rank_list = []
    avg_pos_cnt = 0
    avg_med_cnt = 0
    avg_neg_cnt = 0
    for i,sorted_ix in enumerate(sorted_ix_list):
        not_recall = []
        not_recall_score = []
        rank = []
        med_cnt = 0
        pos_cnt = 0
        neg_cnt = 0
        for j,label in enumerate(labels[i]):
            if int(label) == 1:
                pos_cnt += 1
                if j not in sorted_ix[:top_k]:
                    rank.append(list(sorted_ix).index(j))
                    not_recall.append(j)
                    not_recall_score.append(score_list[i][j])

            if int(label) == 2:
                med_cnt += 1
            if int(label) == 0:
                neg_cnt += 1
        avg_pos_cnt += pos_cnt
        avg_med_cnt += med_cnt
        avg_neg_cnt += neg_cnt
        wrong_recall = []
        wrong_recall_score = []

        total_cnt = 0
        pos_recalled_cnt = 0
        med_recalled_cnt = 0
        for ele in sorted_ix[:top_k]:
            total_cnt += 1
            label = int(labels[i][ele])
            if int(label) == 2:
                med_recalled_cnt += 1
            if int(label) == 1:
                pos_recalled_cnt += 1
            if int(label) == 0:      
                wrong_recall.append(ele)
                wrong_recall_score.append(score_list[i][ele])
        rank_list.append(rank.copy())
        not_recall_list.append(not_recall.copy())
        wrong_recall_list.append(wrong_recall.copy())
        not_recall_score_list.append(not_recall_score.copy())
        wrong_recall_score_list.append(wrong_recall_score.copy())

        pos_cnt_list.append(pos_cnt)
        cnt_list.append(len(labels[i]))
        if pos_cnt == 0:
            continue
        pos_recalled_ratio = float(pos_recalled_cnt)/pos_cnt
        pos_recalled_ratio_list.append(pos_recalled_ratio)
    return pos_recalled_ratio_list,float(avg_pos_cnt)/len(sorted_ix_list),float(avg_neg_cnt)/len(sorted_ix_list),not_recall_list,wrong_recall_list,rank_list,pos_cnt_list,cnt_list,not_recall_score_list,wrong_recall_score_list

def main():
    csv_file_path = 'multi_doc_test_qa_data_805.csv'
    #csv_file_path = '大海捞针数据集评测集_all.csv'

    df = pd.read_csv(csv_file_path)
    dic = {}
    embedding_dic = {}
    pos_cnt_dic = {}
    neg_cnt_dic = {}

    cnt = 0
    black_list = ['总结该篇报告的提纲。提纲需分为三级。']
    start_time = time.time()
    df = df[:]
    for index, row in df.iterrows():
        the_id = row['id']
        label = row['final_label']
        page_content = row['text']
        page_content = page_content.replace('https://mp.weixin.qq.com/s/lqurfsdbn6x9ocSyBxSO5A','')
        page_content = page_content.replace('https://ﬁnance.sina.com.cn/zl/china/2023-08-21/zl-imzhxswf2688110.shtml','')
        page_content = page_content.replace('report＠wind.com.cn','')
        question = row['question']
        if question in black_list:
            continue
        query = ''
        rationale = ''
        if int(label) == 1:
            if question in pos_cnt_dic:
                if pos_cnt_dic[question] > 50:
                    continue
                pos_cnt_dic[question] += 1
            else:
                pos_cnt_dic[question] = 1

        if int(label) == 0:
            if question in neg_cnt_dic:
                neg_cnt_dic[question] += 1
            else:
                neg_cnt_dic[question] = 1
        cnt += 1
        ele = [page_content,label]
        if question in dic:
            dic[question].append(ele.copy())
        else:
            dic[question] = [query,rationale,ele.copy()]
    labels = get_label(dic)
    dic = {key: value for key, value in dic.items() if (key not in pos_cnt_dic or pos_cnt_dic[key] > 4)}
###################################################cos_sim_max_seg
    print('running rank_by_cos_max_seg_sim')
    cos_max_seg_sorted_ix_list,labels,score_list = rank_by_cos_max_seg_sim(dic)
    print('time_cost',time.time() - start_time)

    ################################################### res

    cal_roc(score_list,labels)

    topk_list = [5,10,20,50,60,120,240]
    for topk in topk_list:
        print('running_topk:',str(topk))
        print('cos_max_seg')
        print('cos_max_seg_sorted_ix_list',len(cos_max_seg_sorted_ix_list))
        print('labels',len(labels))
        pos_recalled_ratio_list,avg_pos_cnt,avg_neg_cnt,not_recall_list,wrong_recall_list,rank_list,pos_cnt_list,cnt_list,not_recall_score_list,wrong_recall_score_list = cal_rank_res(topk,cos_max_seg_sorted_ix_list,labels,score_list)
        print('not_recall_list',not_recall_list)

if __name__ == "__main__":
    main()
