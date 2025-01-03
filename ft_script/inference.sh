export CUDA_VISIBLE_DEVICES=1
model_task=snrft_mode0
model_dir=/hpc2hdd/home/simonsyguo/whd/NextX-FT/output_ckpt/snrft_mode0_ep5_lambda0.01/checkpoint-699
test_data_dir=data/GSM8k/test.jsonl
result_output_dir=eval_result/${model_task}.jsonl
python inference_loop.py $model_dir $test_data_dir $result_output_dir > logs/eval_${model_task}_bz32_ckptep3.log 2>&1 &