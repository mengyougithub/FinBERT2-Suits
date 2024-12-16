import csv
from collections import defaultdict
import os
from datetime import datetime
def returndatestr():
	# 获取当前日期和时间
	now = datetime.now()

	date_for_filename = now.strftime("%m%d")
	return date_for_filename
# 初始化一个字典来存储每个模型的F1值和对应的行数


model_f1_scores = defaultdict(list)


os.chdir("finetune_downstreamtask")

topk=1
 
def summary(input_filename,output_filename,writemode="a"):
	with open(input_filename, mode='r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			try:
				model_name = row['experimentname']
				f1_score = float(row['f1'])
				# 确保模型名称对应的列表存在，否则创建
				if model_name not in model_f1_scores:
					model_f1_scores[model_name] = []

				# 将F1分数和行号添加到模型的列表中
				model_f1_scores[model_name].append((f1_score, reader.line_num,row))
			except :
				# 处理实验名称或F1分数缺失的情况
				pass
	with open(output_filename, writemode, newline='') as csvfile:
		fieldnames = list(row.keys())
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		# 对每个模型的F1分数进行排序，并挑选出最高的两个
		for model in model_f1_scores:
			sorted_scores = sorted(model_f1_scores[model], key=lambda x: x[0], reverse=True)
			topk_scores = sorted_scores[:topk]
			# 打印模型名称和对应的最高两个F1分数及行号
			# print(f"Model: {model}")
			for score, line_num,row in topk_scores:
				# print(f"  F1 Score: {score} at Line: {line_num},row:{row}")
				writer.writerow(row)	   

def summaryall():
	# List of filenames to process
	output_filename ="all_summary_"+ returndatestr()+".csv"

	print(output_filename)
	filenames = [
		'industry.csv',
		'ner.csv', 
		'sentiment.csv',
		'sentiment.csv'
	]

	# Process each file
	for filename in filenames:
		summary(filename,output_filename,writemode="w")

#summary one file
input_filename ='finetune_downstreamtask/sentiment2.csv'
output_filename =input_filename.split("/")[-1].split(".")[0]+"_summary.csv"
summary(input_filename,output_filename,writemode="a")


#summary all file
summaryall()