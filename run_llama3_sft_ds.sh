cur_data=$(date +"%Y-%m-%d")
cur_time=$(date +"%H-%M-%S")
echo $cur_time

log_dir=logs/$cur_data
output_dir=sft_output/llama3/$cur_data/$cur_time

if [ -d "$log_dir" ]; then
    echo "log_dir $log_dir already exists."
else
    mkdir -p "$log_dir"
    echo "log_dir $log_dir created."
fi

if [ -d "$output_dir" ]; then
    echo "log_dir $output_dir already exists."
else
    mkdir -p "$output_dir"
    echo "log_dir $output_dir created."
fi


# debug
# export CUDA_VISIBLE_DEVICES=0
# python sft_model.py sft_config/llama3/config_r256_bz4_stack_exchange_paired.json $output_dir
gpus=0,1,2,3
export NCCL_P2P_DISABLE=1
deepspeed --include=localhost:$gpus sft_llama3.py \
    --deepspeed="deepspeed_config/ds_config_zero2.json"\
    --output_dir=$output_dir \
    --max_steps=1000 \
    --logging_steps=10 \
    --save_steps=10 \
    --save_total_limit=2 \
    --template_str_dir="./formatting_template/Llama3.yaml" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --remove_unused_columns=False \
    --run_name="sft_llama3_another_version" \
    # --report_to="wandb"