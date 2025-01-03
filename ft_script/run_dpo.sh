export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5
data_name=GSM8k
task=dpo
epochs=3
root_dir=/home/wuhaodong
torchrun --nproc_per_node=2 dpo.py \
    --model_name_or_path ../sft_output/sft_mode0/checkpoint-348 \
    --data_path ../data/${data_name}/dpo.json \
    --bf16 True \
    --output_dir ../sft_output/${task}_dataset${data_name} \
    --num_train_epochs $epochs \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit $epochs \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --remove_unused_columns False \
    --content_field prompt \
    --reject_field reject \
    --chosen_field accept \
    --ft_type $task \
    --if_lora True \
    --lora_r 256 \
    --lora_alpha 512 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True #> ../logs/${task}_mode_${mode}_dataset_${data_name}.log 2>&1 &

    