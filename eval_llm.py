from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader


# 配置
def get_rouge_score(hyps, refs):
    import rouge

    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    py_rouge_scores = evaluator.get_scores(hyps, refs)
    return py_rouge_scores


model_name_or_path = (
    "../ckpts/meta-llama/Meta-Llama-3-8B-Instruct"  # 替换为你微调后的模型路径
)
peft_model_path = "sft_output/sft_samsum_llama3"  # 替换为你使用 PEFT 微调的模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_dataset = load_dataset(
    "json",
    data_files="/hpc2hdd/home/sguo349/whd/hf_datasets/Samsung/samsum/data/test.json",
    split="train",
)
# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
# # 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
base_model.to(device)

# # 如果使用了 PEFT 微调，加载微调模型
if peft_model_path:
    model = PeftModel.from_pretrained(base_model, peft_model_path)
else:
    model = base_model

model.to(device)


# 将 Hugging Face Dataset 转换为 PyTorch DataLoader
dataloader = DataLoader(eval_dataset, batch_size=16)

# 输入测试
sys_prompt = "Summarize this dialogue: "
labels = []
preds = []
for batch in tqdm(dataloader, desc="generating"):
    labels.extend(batch["summary"])

    messages = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": it},
        ]
        for it in batch["dialogue"]
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, padding=True, return_tensors="pt").to(device)

    # 文本生成
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=50,  # 设置生成文本的最大长度
        num_return_sequences=1,  # 生成的序列数量
        repetition_penalty=1.5,
        no_repeat_ngram_size=5,
        temperature=0.9,  # 控制生成的多样性
        top_k=100,  # 通过设置较小的top_k，可以避免生成低概率的词
        top_p=0.95,  # 通过核采样(p-nucleus sampling)来控制生成的多样性
    )

    # 解码生成的文本
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    generated_text = [i.split("assistant\n\n")[-1] for i in generated_text]
    preds.extend(generated_text)
    print(f"Sample1: {generated_text[0]}")
metrics_result = get_rouge_score(refs=labels, hyps=preds)
print(metrics_result)
