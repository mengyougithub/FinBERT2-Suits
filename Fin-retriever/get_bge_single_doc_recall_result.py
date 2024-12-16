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
model_path = './roberta_all_0724_mlm_optimized_1-5epoch_checkpoint-60441_encoder_model_test_400_600_5w_rm_dup_hn_20_range_0_200_eval_pos_neg_select_4w_add_general_shuf_warmup_ratio_01/checkpoint-1200/'

model = SentenceTransformer(model_path)
print('model',model)
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


def get_m3e_embedding(content_list):
    res = model.encode(content_list)
    return res


def compute_sim_score(query,doc_list):
    query = '为这个句子生成表示以用于检索相关文章：' + query
    embeddings = get_m3e_embedding([query] + doc_list)
    question_embedding = embeddings[0]
    question_embedding = question_embedding/np.sqrt((question_embedding**2).sum())
    cos_sim_list = []
    for embedding in embeddings[1:]:
        content_embedding = embedding/np.sqrt((embedding**2).sum())
        cos_sim = dot(question_embedding, content_embedding)
        cos_sim_list.append(cos_sim)
    return cos_sim_list


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
    sorted_ix_list = []
    score_list = []
    labels = []
    count = 0
    for question,page_contents in dic.items():
        count += 1
        if count % 10 == 0:
            print('rank_by_cos_max_seg_sim_done_with ',str(count))
        max_seg_cos_sim = []
        label = []
        query = page_contents[0]
        rationale = page_contents[1]
        for page_content in page_contents[2:]:
            label.append(page_content[1])
            max_cos_sim0 = []
            max_cos_sim1 = []
            max_cos_sim2 = []
            max_cos_sim3 = []
            text_list = []
            for i in range(1+(len(page_content[0]))//shift):
                ele = page_content[0][(0 + i*shift):(400 + i*shift)]
                if len(ele) == 0:
                    continue
                text_list.append(ele)
            max_cos_sim0 = compute_sim_score(question,text_list) 
            max_seg_cos_sim.append(max(max_cos_sim0))    

        max_seg_cos_sim_new = []
        for ele in max_seg_cos_sim:
            max_seg_cos_sim_new.append(float(ele))
        labels.append(label.copy())
        sorted_ix = np.argsort(-np.array(max_seg_cos_sim_new))
        sorted_ix_list.append(sorted_ix)
        score_list.append(max_seg_cos_sim_new[:])
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




def remove_stop_word(words):
    new_words = []
    #print('words',words)
    for word in words:
        if word in stop_words:
            continue
        chn_res = re.findall(u'[\u4e00-\u9fa5]',word)
        eng_res = re.findall(r'[a-zA-Z]+', word)
        num_res = re.findall(r'\d+', word)
        if len(chn_res) == 0 and len(eng_res) <= 1 and len(num_res) == 0:
            continue
        new_words.append(word)
    return new_words

def cal_roc(score_list,labels):
    thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.999]
    #specific_thresholds = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98]

    score_list = np.concatenate(score_list).tolist()
    label = np.concatenate(labels).tolist()

    print('score_list',score_list[:100])
    print('label',label[:100])

    precision_results, recall_results, accuracy_results, f1_results = calculate_metrics_for_thresholds(np.array(label), np.array(score_list),thresholds)
    for threshold in thresholds:
        print(f"Threshold: {threshold} ",f" Precision: {precision_results[thresholds.index(threshold)]:.2f}",f" Recall: {recall_results[thresholds.index(threshold)]:.2f}",f" Accuracy: {accuracy_results[thresholds.index(threshold)]:.2f}",f" F1 Score: {f1_results[thresholds.index(threshold)]:.2f}")
    fpr, tpr, thresholds = roc_curve(label, score_list)
    roc_auc = auc(fpr, tpr)
    print('roc_auc',roc_auc)

def cal_rouge(text0,text1):
    text0 = text0.lower()
    text1 = text1.lower()
    text0_cut = jieba.cut(text0)
    text0_cut_rm_stop_word = remove_stop_word(text0_cut)
    text1_cut = jieba.cut(text1)
    text1_cut_rm_stop_word = remove_stop_word(text1_cut)
    cnt = 0
    common_word = []
    for ele in text0_cut_rm_stop_word:
        if ele in text1_cut_rm_stop_word:
            common_word.append(ele)
            cnt += 1

    return cnt/len(text0_cut_rm_stop_word),common_word

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
        #print('sorted_ix',sorted_ix)
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
    csv_file_path = 'relevance_testset_kimi_GPT4_0709_final_have_positive.csv'

    df = pd.read_csv(csv_file_path)
    dic = {}
    pos_cnt_dic = {}
    neg_cnt_dic = {}
    correct_dict = {10:0,22:1,1240:1,8:1,9:1,38:1,39:1,42:1,46:1,47:1,50:1,53:0,55:0,62:1,63:1,64:1,65:1,66:1,67:1,68:1,69:0,75:1,78:0,79:0,84:0,150:1,166:1,189:1,199:1,347:1,376:1,406:1,479:1,496:1,456:1,451:1,1013:0,1143:0,1147:0,1153:0,1055:0,1162:0,1171:0,1087:0,1218:1,1222:1,1238:1,1375:1,1378:0,1414:1,1421:1,1424:0,1425:0,1427:0,1428:0,1429:1,1454:0,1460:0,\
                        1480:0,1483:0,1485:1,1499:1,1004:0,1049:0,1042:0,1072:0,1078:0,1075:0,1430:1,1369:1,1363:1,1361:1,1144:1,1101:1,1101:1}
    cnt = 0
    black_list = ['总结该篇报告的提纲。提纲需分为三级。']
    start_time = time.time()
    #df = df[:20]
    for index, row in df.iterrows():
        the_id = row['id']
        org_label = row['label']
        label = row['final_label']
        if int(label) == 1:
            continue
        if int(label) == 2:
            label = 1
        page_content = row['page_content']
        page_content = page_content.replace('https://mp.weixin.qq.com/s/lqurfsdbn6x9ocSyBxSO5A','')
        page_content = page_content.replace('https://ﬁnance.sina.com.cn/zl/china/2023-08-21/zl-imzhxswf2688110.shtml','')
        page_content = page_content.replace('report＠wind.com.cn','')

        question = row['question']
        if question in black_list:
            continue
        query = row['query']
        rationale = row['rationale']
        if int(label) == 1:
            if query in pos_cnt_dic:
                if pos_cnt_dic[query] > 100:
                    continue
                pos_cnt_dic[query] += 1
            else:
                pos_cnt_dic[query] = 1

        if int(label) == 0:
            if query in neg_cnt_dic:
                if neg_cnt_dic[query] > 100:
                    continue
                neg_cnt_dic[query] += 1
            else:
                neg_cnt_dic[query] = 1
        cnt += 1
        if query == '' or question == '' or rationale == '':
            continue
        ele = [page_content,label]
        if question in dic:
            dic[question].append(ele.copy())
        else:
            dic[question] = [query,rationale,ele.copy()]
    labels = get_label(dic)
###################################################cos_sim_max_seg
    print('running rank_by_cos_max_seg_sim')
    cos_max_seg_sorted_ix_list,labels,score_list = rank_by_cos_max_seg_sim(dic)
    print('time_cost',time.time() - start_time)
    ################################################### res

    cal_roc(score_list,labels)

    topk_list = [1,3,5,10]
    for topk in topk_list:
        print('running_topk:',str(topk))
        pos_recalled_ratio_list,avg_pos_cnt,avg_neg_cnt,not_recall_list,wrong_recall_list,rank_list,pos_cnt_list,cnt_list,not_recall_score_list,wrong_recall_score_list = cal_rank_res(topk,cos_max_seg_sorted_ix_list,labels,score_list)
        print('not_recall_list',not_recall_list)
        print('avg_pos_cnt',avg_pos_cnt)
        print('avg_neg_cnt',avg_neg_cnt)
        print('pos_recalled_ratio',float(sum(pos_recalled_ratio_list))/len(pos_recalled_ratio_list))


if __name__ == "__main__":
    main()
