cur_data=$(date +"%Y-%m-%d")
cur_time=$(date +"%H-%M-%S")
echo $cur_time

log_dir=logs/$cur_data
output_dir=sft_output/Qwen/$cur_data/$cur_time

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
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node=4 --master-port=29502 sft_model.py sft_config/qwen/config_r256_bz4_stack_exchange_paired.json $output_dir > $log_dir/$cur_time.log 2>&1 &
# torchrun --nproc_per_node=4 sft_model.py sft_config/config_r256_bz4.json $output_dir