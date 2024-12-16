import csv
import torch
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification,BertTokenizer
from transformers import AutoModelForTokenClassification,BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
modelpath=""
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = AutoModelForSequenceClassification.from_pretrained(modelpath).to(device)

def infer_batch_sequencecls(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        softmax_probs = F.softmax(logits, dim=-1)
        # BERT中的第一个标记（[CLS]）的索引为0
        cls_index = 0
        # 获取BERT模型的输出
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 获取第n层的隐藏状态
        n=12
        nlayers_states = outputs.hidden_states[n]
        # 提取CLS标记的向量
        cls_vectors = nlayers_states[:, cls_index, :].detach().cpu().numpy()    
        predicted_class_ids = logits.argmax(dim=-1).tolist()
    id2label = {0: '负面', 1: '正面'}
    predicted_classes = [id2label[pred] for pred in predicted_class_ids]
    # 返回预测类别、softmax概率和CLS向量
    return predicted_classes, softmax_probs.tolist(), cls_vectors.tolist()

# # 示例文本
# texts = ["这是一个正面的评论", "这是一个负面的评论"]
# # 调用函数
# predicted_classes, softmax_probs, cls_vectors = infer_batch_sequencecls(texts)

texts =  ['稳定均衡定义和特点, 纳什均衡定义和特点','稳定均衡与纳什均衡的区别和联系','稳定均衡与纳什均衡的区别和联系','稳定均衡与纳什均衡的区别和联系']
infer_batch_sequencecls(texts)
