from transformers import AutoTokenizer,DataCollatorForTokenClassification,AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import os
from downstream_NER_dataset import companyNERDataset,save_dataset_to_csv
import torch
import argparse
import random
import csv
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6" 
parser = argparse.ArgumentParser(description='Model Path Argument')
parser.add_argument('--model_name', type=str, required=False, help='Path to the model directory')
parser.add_argument('--dataset', type=str, required=False,default="company", help='dataset name')
args = parser.parse_args()
model_name = args.model_name
dataset = args.dataset
print(model_name,dataset)

if not model_name:
    model_name = "hfl/chinese-roberta-wwm-ext"


if dataset=="name":
    file_path1 = 'downstream_data/train_name.txt'
    file_path2 = 'downstream_data/test_name.txt'

elif dataset=="company":
    file_path2= 'downstream_data/test_company.txt'
    file_path1 = 'downstream_data/train_company.txt'
else:
    file_path2= 'downstream_data/test_company.txt'
    file_path1 = 'downstream_data/train_company.txt'

    
if model_name.split("/")[-1]=="encoder_model":
    experimentname=model_name.split("/")[-2]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
else:
    experimentname=model_name.split("/")[-1]+"_"+os.path.basename(file_path1).split("_")[1].split(".")[0]
save_directory='classifier_models/'+experimentname
csv_file_path="ner_0627.csv"
metric_name="f1"
   
train_dataset = companyNERDataset(file_path1)
test_dataset = companyNERDataset(file_path2)
print(len(train_dataset))
print(len(test_dataset))
label2id , id2label =train_dataset.getlabel2id()
label_list = sorted(label2id, key=label2id.get, reverse=False)# 按值的数值大小进行排序 输出排序后的键列表
# {'O': 2, 'B-PER': 0, 'I-PER': 1}
print(label2id)
print(label_list)  
# ['B-PER','I-PER','O']


training_args = TrainingArguments(
    output_dir=save_directory,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=10000,
    save_strategy = "steps",
    # eval_steps=20 ,
    eval_steps=10 if dataset=="name" else 50,
    evaluation_strategy="steps",
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
def tokenize_and_align_labels2(dataset):
    tokenized_records=[]
    for record in dataset.records:
        
        # tokenized_input = tokenizer("".join(record["tokens"]), truncation=True,max_length=128, is_split_into_words=True)
        tokenized_input = tokenizer(record["tokens"], truncation=True,max_length=512, is_split_into_words=True)
        # word_ids = tokenized_input.word_ids(batch_index=i)
        label=[label2id[label] for label in  record["ner_tags"] ]
        # tokenized_input["labels"] = labels
        
        word_ids = tokenized_input.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_input["labels"] = label_ids

        # labels.append(label_ids)
        tokenized_records.append(tokenized_input)
    dataset.records=tokenized_records
    return dataset

tokenized_train_dataset=tokenize_and_align_labels2(train_dataset)
tokenized_test_dataset=tokenize_and_align_labels2(test_dataset)


seqeval = evaluate.load("seqeval")
evalnum=0
eval_steps=training_args.eval_steps
def compute_metrics(p):#p=EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    global evalnum
    evalnum+=1
    global eval_steps
    currentsteps=evalnum*eval_steps
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    metrics={
        'experimentname':experimentname,
        'currentsteps':currentsteps,
        "accuracy": results["overall_accuracy"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

    if training_args.local_rank in (0, -1):
        # 检查文件是否存在
        file_exists = os.path.isfile(csv_file_path)
        # 准备字段名列表，包括新增的model和epoch
        fieldnames = ['experimentname', 'currentsteps', 'accuracy','precision', 'recall', 'f1']

        # 写入CSV文件
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 只在文件不存在时写入标题行
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    return metrics

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id
)

if dataset=="company":
    tokenized_test_dataset=tokenized_test_dataset[0:300]   #减少评测集大小，加快评测速度

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,    
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
