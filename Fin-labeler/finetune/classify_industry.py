import torch
import csv
from downstream_dataset import *
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader, SequentialSampler,RandomSampler
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
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
parser.add_argument('--model_name', type=str, required=False, help='Path to the model directory')
args = parser.parse_args()
model_name = args.model_name
print(model_name)

if not model_name:
    model_name = "hfl/chinese-roberta-wwm-ext"
    
    
file_path1= "downstream_data/train_industry_cla.txt" 
train_dataset=industryDataset(file_path1)
file_path2= "downstream_data/test_industry_cla.txt" 
test_dataset=industryDataset(file_path2)

print("train_dataset_len:",len(train_dataset))
print("test_dataset_len:",len(test_dataset))
label2id , id2label=train_dataset.getlabel2id()
print(id2label)
label2id , id2label=test_dataset.getlabel2id()
print(id2label.values())
print(len(id2label.values()))

train_dataset=train_dataset[0:4000]
test_dataset=test_dataset[0:400]

#28分类 {3: '医药', 2: '农林牧渔', 5: '国防军工', 4: '商贸零售', 10: '房地产', 16: '电力设备', 9: '建筑', 25: '非银行金融', 1: '传媒', 20: '计算机', 13: '汽车', 17: '电子元器件', 8: '建材', 21: '轻工制造', 7: '家电', 12: '机械', 22: '通信', 19: '纺织服装', 24: '银行', 26: '食品饮料', 15: '电力及公用事业', 27: '餐饮旅游', 0: '交通运输', 11: '有色金属', 14: '煤炭', 6: '基础化工', 18: '石油石化', 23: '钢铁'
num_classes = len(id2label) 
metric_name = "f1"
if model_name.split("/")[-1]=="encoder_model":
    experimentname=model_name.split("/")[-2]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
else:
    experimentname=model_name.split("/")[-1]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
save_directory='classifier_models/'+experimentname
csv_file_path="industry_0627.csv"

training_args = TrainingArguments(
    output_dir=save_directory,
    num_train_epochs=1,
    per_device_train_batch_size=5,
    # per_device_eval_batch_size=16,
    logging_steps=1,
    save_steps=20000,
    save_strategy = "steps",
    eval_steps=10,
    evaluation_strategy="steps",
    # learning_rate=2e-5,
    learning_rate=5e-5,
    weight_decay=0.01,
    # load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

evalnum=0
eval_steps=training_args.eval_steps

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, problem_type="single_label_classification", num_labels=num_classes)
data_collator = industry_collator(tokenizer=tokenizer, label2id=label2id, max_length=510)

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
        file_exists = os.path.isfile(csv_file_path)
        
        fieldnames = ['experimentname', 'currentsteps', 'accuracy', 'precision', 'recall', 'f1']

        # 写入CSV文件
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 只在文件不存在时写入标题行
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

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
# model.save_pretrained(save_directory)


