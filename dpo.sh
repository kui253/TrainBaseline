hf_home=../ckpts
hf_dataset_home=../hf_datasets

cur_data=$(date +"%Y-%m-%d")
cur_time=$(date +"%H-%M-%S")
echo $cur_time
export CUDA_VISIBLE_DEVICES=4,5
log_dir=logs/dpo_Qwen/$cur_data
output_dir=dpo_output/Qwen/$cur_data/$cur_time

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
export NCCL_P2P_DISABLE=1
accelerate launch --num_processes=2 --config_file multi_gpu.yaml run_dpo.py \
    --dataset_name $hf_dataset_home/trl-lib/ultrafeedback_binarized \
    --model_name_or_path ../ckpts/Qwen/Qwen2-7B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir $output_dir \
    --no_remove_unused_columns \
    --use_peft \
    --dataset_num_proc 16 \
    --lora_r 32 \
    --lora_alpha 16 #> $log_dir/$cur_time.log 2>&1 &