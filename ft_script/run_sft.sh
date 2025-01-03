export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5
data_name=GSM8k
task=sft
run_name=lora_embed_norm_all_linear_initial
epochs=3
root_dir=/home/wuhaodong
torchrun --nproc_per_node=2 --master_port=29500 sft.py \
    --model_name_or_path ${root_dir}/ckpts/meta-llama/Llama-2-7b-hf \
    --data_path ../data/${data_name}/train.json \
    --bf16 True \
    --output_dir ../sft_output/${task}_dataset${data_name}_$run_name \
    --num_train_epochs $epochs \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit $epochs \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --run_name $run_name \
    --if_embed_norm True \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --ft_mode $task \
    --if_lora True \
    --lora_r 256 \
    --lora_alpha 512 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True > ../logs/${task}_dataset_${data_name}_$run_name.log 2>&1 &

    