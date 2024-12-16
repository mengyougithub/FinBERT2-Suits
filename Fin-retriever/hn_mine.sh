export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path ./roberta_all_0724_mlm_optimized_1-5epoch/checkpoint-60441/encoder_model \
--input_file ./test_400_600_5w_rm_dup.json \
--output_file test_400_600_5w_rm_dup_hn_20_range_0_50.json \
--range_for_sampling 0-50 \
--query_instruction_for_retrieval "" \
--negative_number 50 \
--use_gpu_for_searching 
#--model_name_or_path ./roberta_all_0724_mlm_optimized_1-5epoch/checkpoint-60441/encoder_model/ \
