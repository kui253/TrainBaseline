from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import yaml
from datasets import load_dataset
from argparse import ArgumentParser
import random
from tqdm import tqdm


def main(args):
    DIR2NAME = {
        "mathqa": "../hf_datasets/allenai/math_qa/math_qa.py",
        "pubmedqa": "../hf_datasets/bigbio/pubmed_qa/pubmed_qa.py",
        "agieval": "../hf_datasets/agieval/agieval.py",
        "winogrande": "../hf_datasets/winogrande/winogrande.py",
        "arc_easy": "../hf_datasets/allenai/ai2_arc/ARC-Easy",
        "arc_challenge": "../hf_datasets/allenai/ai2_arc/ARC-Challenge",
    }
    SYS_PROMPT4DS = {
        # "mathqa": "Solve the given math problem. Do not explain the steps and only output your choice in the format: Answer: a/b/c/d/e. Do not output any other information",
        "mathqa": "Solve the given math problem, and output the rationale and your choice in the format: Rationale: ... \n Answer: a/b/c/d/e. Do not output any other information",
        "pubmedqa": "Read the meterials, and anwser the corresponding question. Output your choice in the format:Explanation: ... \n Answer: yes/no. Do not output any other information",
        "agieval": "Read the meterials, and anwser the corresponding question. Output your choice in the format: Answer: A/B/C/D/E. Do not output any other information",
        "winogrande": "Choose the correct word to fill in the underlined parts (marked by _) of a given sentence, output your choice in the format: Answer: 1/2. Do not output any other information",
        "arc_easy": "Choose the correct option for the given question, Output your choice in the format: Answer: A/B/C/D. Do not output any other information",
        "arc_challenge": "Choose the correct option for the given question, Output your choice in the format: Answer: A/B/C/D. Do not output any other information",
    }
    # 配置
    # eval_dataset = load_dataset(DIR2NAME[args.sys_prompt])["test"]
    # index_sample = random.sample(list(range(len(eval_dataset))), args.max_sample)

    model_name_or_path = (
        args.base_model_dir
    )  # "../ckpts/Qwen/Qwen2-7B-Instruct"  # 替换为你微调后的模型路径
    peft_model_path = args.peft_model_dir
    # (
    #     "./sft_output/Qwen/2024-10-07/10-20-11-qwen"  # 替换为你使用 PEFT 微调的模型路径
    # )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.chat_template_dir is not None:
        with open(args.chat_template_dir, "r") as f:
            cfg = yaml.safe_load(f)
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    base_model.to(device)

    # 如果使用了 PEFT 微调，加载微调模型
    if peft_model_path:
        model = PeftModel.from_pretrained(base_model, peft_model_path)
    else:
        model = base_model

    model.to(device)

    # 输入测试
    sys_prompt = SYS_PROMPT4DS[args.sys_prompt]
    if args.interact:
        while True:
            prompt = input("User: ")
            messages = [
                {"role": "system", "content": "Answer the questions."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=(
                    cfg["instruction_template"]
                    if args.chat_template_dir is not None
                    else None
                ),
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            # 文本生成
            output_ids = model.generate(
                input_ids,
                max_length=500,  # 设置生成文本的最大长度
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,  # 生成的序列数量
                temperature=0.7,  # 控制生成的多样性
                top_k=50,  # 通过设置较小的top_k，可以避免生成低概率的词
                top_p=0.95,  # 通过核采样(p-nucleus sampling)来控制生成的多样性
                repetition_penalty=1.3
            )

            # 解码生成的文本
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print("*" * 30)
            print(f"Agent: {generated_text}")

    else:
        for n in tqdm(index_sample):
            item = eval_dataset[n]
            prompt = item["Problem"] + "\n" + item["options"]
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            c_a = item["correct"]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=(
                    cfg["instruction_template"]
                    if args.chat_template_dir is not None
                    else None
                ),
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            # 文本生成
            output_ids = model.generate(
                input_ids,
                max_length=400,  # 设置生成文本的最大长度
                num_return_sequences=1,  # 生成的序列数量
                temperature=0.7,  # 控制生成的多样性
                top_k=50,  # 通过设置较小的top_k，可以避免生成低概率的词
                top_p=0.95,  # 通过核采样(p-nucleus sampling)来控制生成的多样性
            )

            # 解码生成的文本
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("*" * 30)
            print(f"Agent: {generated_text}")
            print(f"correct answer: {c_a}")
            print("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=None,
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--peft_model_dir",
        type=str,
        default=None,
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--chat_template_dir",
        type=str,
        default=None,
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--sys_prompt",
        type=str,
        default=None,
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--interact",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=20,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    main(parser.parse_args())
