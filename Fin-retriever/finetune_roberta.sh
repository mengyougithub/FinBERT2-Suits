export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 \
--master-port 18731 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir roberta_all_0724_mlm_optimized_1-5epoch_checkpoint-60441_encoder_model_test_400_600_5w_rm_dup_hn_20_range_0_200_eval_pos_neg_select_4w_add_general_shuf_warmup_ratio_01_group_15 \
--model_name_or_path ./roberta_all_0724_mlm_optimized_1-5epoch/checkpoint-60441/encoder_model \
--train_data ./finetune_traindata_sample.json \
--save_only_model True \
--learning_rate 2e-5 \
--bf16 \
--num_train_epochs 5 \
--save_steps 100 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--dataloader_drop_last True \
--normlized True \
--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章：" \
--train_group_size 15 \
--temperature 0.01 \
--query_max_len 512 \
--passage_max_len 512 \
--warmup_ratio 0.1 \
--weight_decay 0.03 \
--logging_steps 200 

#--model_name_or_path ./roberta_all_mlm_lr5_encoder_model \
#--model_name_or_path ./roberta_all_0724_mlm_optimized_1-5epoch/checkpoint-60441/encoder_model \
#--model_name_or_path roberta_7.3b_optimized \
#--model_name_or_path ./roberta_all_0724_mlm_optimized_1-5epoch/checkpoint-60441/encoder_model \

