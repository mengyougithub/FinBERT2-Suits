run_model_classification() {
    local model_name="$1"
    local CUDA_VISIBLE_DEVICES="1"
    local master_port="29509"

    # 舆情分析
    echo "Running sentiment analysis model..."
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node 1 --master-port "${master_port}" classify_financialnews_sentiment.py --model_name "${model_name}"

    # 情感分析
    echo "Running sentiment classification model..."
     CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node 1 --master-port "${master_port}" classify_sentiment.py --model_name "${model_name}"

    # 行业分类
    echo "Running industry classification model..."
     CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node 1 --master-port "${master_port}" classify_industry.py --model_name "${model_name}"

    # 命名实体识别 (NER) - 根据不同数据集
    local datasets=("name" "company")
    for dataset in "${datasets[@]}"; do
        echo "Running NER classification model for dataset: ${dataset}..."
         CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"  torchrun --nproc_per_node 1 --master-port "${master_port}" classify_NER.py --model_name "${model_name}" --dataset "${dataset}"
    done
}



# run_model_classification "google-bert/bert-base-chinese"


