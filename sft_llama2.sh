cur_data=$(date +"%Y-%m-%d")
cur_time=$(date +"%H-%M-%S")
echo $cur_time

log_dir=logs/$cur_data
output_dir=sft_output/llama2/$cur_data/$cur_time

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
export CUDA_VISIBLE_DEVICES=0,1,2,4
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node=4 sft_model.py sft_config/llama2/config_r64_bz128_nocausal_ep3.json $output_dir > $log_dir/$cur_time.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
# 
# torchrun --nproc_per_node=6 sft_model.py sft_config/llama3/config_r256_bz8_pubmedqa.json $output_dir > $log_dir/$cur_time.log 2>&1 &
# torchrun --nproc_per_node=4 sft_model.py sft_config/llama3/config_r256_bz4_stack_exchange_paired.json $output_dir > $log_dir/$cur_time.log 2>&1 &
# torchrun --nproc_per_node=4 sft_model.py sft_config/config_r256_bz4.json $output_dir

# 下次使用deepspeed启动
# gpus=0,1,2,3
# export NCCL_P2P_DISABLE=1
# deepspeed --include=localhost:$gpus sft_model.py \
#     --deepspeed="deepspeed_config/ds_config_zero2.json" #> $log_dir/$cur_time.log 2>&1 &