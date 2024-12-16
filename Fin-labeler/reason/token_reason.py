from transformers import AutoModelForTokenClassification,BertTokenizer
import torch

modelpath="/root/autodl-tmp/xuxuan/classifier_models/FinBERT_L-12_H-768_A-12_pytorch_company"
tokenizer = BertTokenizer.from_pretrained(modelpath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(modelpath).to(device)

# id2label=model.config.id2label
def infer_batch_tokencls(texts):
    # inputs = tokenizer(texts, return_tensors="pt",truncation=True).to(device)
    inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # 公司NER
    id2label={0: 'B-ORG', 1: 'I-ORG', 2:'O'}
    # 人名NER
    # id2label={0: 'B-PER', 1: 'I-PER', 2:'O'}
    print(id2label)
    predictions = torch.argmax(logits, dim=2)
    predicted_token_classes = [[id2label[t.item()] for t in prediction] for prediction in predictions]
    print(predicted_token_classes)
    return predicted_token_classes   

text1="熵简科技获得了投资者的大力支持"
text2="金山办公的基本情况"
texts = [text1, text2]
infer_batch_tokencls(texts)

