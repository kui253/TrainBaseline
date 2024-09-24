from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 配置
model_name_or_path = (
    "../ckpts/meta-llama/Meta-Llama-3-8B-Instruct"  # 替换为你微调后的模型路径
)
peft_model_path = "sft_output/sft_samsum_llama3"  # 替换为你使用 PEFT 微调的模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"

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
sys_prompt = "Summarize this dialogue: "
while True:
    prompt = input("User: ")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # 文本生成
    output_ids = model.generate(
        input_ids,
        max_length=200,  # 设置生成文本的最大长度
        num_return_sequences=1,  # 生成的序列数量
        temperature=0.7,  # 控制生成的多样性
        top_k=50,  # 通过设置较小的top_k，可以避免生成低概率的词
        top_p=0.95,  # 通过核采样(p-nucleus sampling)来控制生成的多样性
    )

    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Agent: {generated_text}")
