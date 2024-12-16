import csv
import os
from downstream_dataset import *
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import random
import torch

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
setup_seed(42)

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Model Path Argument')
# 添加模型路径参数
parser.add_argument('--model_name', type=str, required=False, help='Path to the model directory')
args = parser.parse_args()

# 从参数中获取模型路径
model_name = args.model_name
print(model_name)
if not model_name:

    model_name = "hfl/chinese-roberta-wwm-ext"

file_path1= "downstream_data/train_FinanceSentimentData.txt" 
train_dataset=SentimentDataset(file_path1)
label2id , id2label=train_dataset.getlabel2id()
train_dataset=train_dataset[0:4000]
file_path2= "downstream_data/test_FinanceSentimentData.txt" 
test_dataset=SentimentDataset(file_path2)
test_dataset=test_dataset[0:400]
num_classes = len(label2id) 

if model_name.split("/")[-1]=="encoder_model":
    experimentname=model_name.split("/")[-2]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
else:
    experimentname=model_name.split("/")[-1]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
output_dir='classifier_models/'+experimentname
csv_file_path = 'sentiment_test.csv'
metric_name = "f1"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=50,
    logging_steps=1,
    save_steps=100000,
    save_strategy = "steps",
    eval_steps=10,
    evaluation_strategy="steps",
    # learning_rate=5e-5,
    learning_rate=5e-5,
    weight_decay=0.01,
    # load_best_model_at_end=True,
    # warmup_steps=500,
    metric_for_best_model=metric_name,
)

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, problem_type="single_label_classification", num_labels=num_classes,id2label=id2label,label2id=label2id)
data_collator = sentiment_collator(tokenizer=tokenizer, max_length=510)

evalnum=0
eval_steps=training_args.eval_steps

def compute_metrics(eval_pred):
    """
    计算并返回评估指标。
    
    参数:
    eval_pred (Tuple[Tensor, Tensor]): 包含预测结果和真实标签的元组。
    """
    # 解包预测结果和真实标签
    predictions, labels = eval_pred
    global evalnum
    evalnum+=1
    # 将预测结果转换为类别
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    
    # 计算评估指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    global eval_steps
    currentsteps=evalnum*eval_steps

    metrics = {
        'experimentname':experimentname,
        'currentsteps':currentsteps,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    if training_args.local_rank in (0, -1):
        # 定义CSV文件名
        # 检查文件是否存在
        file_exists = os.path.isfile(csv_file_path)
        # 准备字段名列表，包括新增的model和epoch
        fieldnames = ['experimentname', 'currentsteps', 'accuracy', 'precision', 'recall', 'f1']

        # 写入CSV文件
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 只在文件不存在时写入标题行
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        # 创建一个包含所有评估指标的字典

    return metrics


# 创建Trainer实例
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    # 添加以下参数
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
