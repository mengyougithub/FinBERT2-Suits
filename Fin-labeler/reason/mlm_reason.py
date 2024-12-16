# Python代码示例，展示如何使用BERT模型进行掩码预测训练
import torch
from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('/data/xuxuan/finbert2/bypy/pretrain/pretrained_model/FinBERT_L-12_H-768_A-12_pytorch')
model = BertForMaskedLM.from_pretrained('/data/xuxuan/finbert2/bypy/pretrain/pretrained_model/FinBERT_L-12_H-768_A-12_pytorch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlminfer(text):
    # text = "“我爱[MASK]"
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # 在序列中随机选择一个位置进行掩码
    mask_index = input_ids.index(tokenizer.mask_token_id)
    # 转换为PyTorch的Tensor格式
    input_tensor = torch.tensor([input_ids])
    # 预测被掩盖的词语
    with torch.no_grad():
        predictions = model(input_tensor)[0]
    # 获取掩码对应的词语的预测概率
    mask_prediction = predictions[0, mask_index].softmax(dim=0)
    # 获取预测概率最高的前5个词语
    top_k = torch.topk(mask_prediction, k=5)
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k.indices.tolist())
    # 输出预测结果
    print("Predicted tokens:", top_k_tokens)
    print("Predicted probabilities:", top_k.values.tolist())
    
# text = "熵简科技股价大跌，股民非常[MASK]"
# mlminfer(text) 

# 提取第n层的CLS向量，对于bert-base，n最大为12
def get_cls_vectors_by_layer(texts, n=12):
    # 对文本进行编码
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # BERT中的第一个标记（[CLS]）的索引为0
        cls_index = 0
        
        # 获取BERT模型的输出
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # 获取第n层的隐藏状态
        nlayers_states = outputs.hidden_states[n]
        
        # 提取CLS标记的向量
        cls_vectors = nlayers_states[:, cls_index, :].detach().cpu().numpy()
    
    # 返回CLS向量列表
    return cls_vectors

# # 假设我们有一个文本列表
# texts = ["Your text here", "Another text example"]
# cls_vectors = get_cls_vectors_by_layer(texts, 1)
# # 打印结果的形状，它应该是 (batch_size, hidden_size)
# print(cls_vectors[0])