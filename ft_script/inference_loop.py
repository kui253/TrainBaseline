from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from typing import Dict
import re
import jsonlines
from tqdm import tqdm
import sys
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
model_dir, test_data_dir, result_output_dir = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir,model_max_length=512,padding_side="right",use_fast=False)
print("model_load")
model.to("cuda")
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
def extract_answer(completion):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

prompt_no_input = PROMPT_DICT["prompt_no_input"]
with jsonlines.open(test_data_dir, 'r') as f:
    test_data = [obj for obj in f]

special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)
    # from peft import PeftModel
    # model = PeftModel.from_pretrained(model, "output_ckpt/lora_ver2")
invalid = 0
correct = 0
writer = jsonlines.open(result_output_dir, 'w')
for item in tqdm(test_data, desc="generating"):
    
    input_data = {"instruction": item['source'][0]}
    input_str = prompt_no_input.format_map(input_data)
    inputs = tokenizer(input_str,return_tensors="pt").input_ids.to("cuda")

    output_ids = model.generate(
        inputs,
        max_new_tokens=256,  
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,  # 生成的序列数量
        num_beams=1
    )

    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    gold_result = extract_answer(item["target"][0])
    pred_result = extract_answer(generated_text)
    result_dict = {}
    result_dict["source"] = item['source'][0]
    result_dict["predict"] = generated_text
    if pred_result == INVALID_ANS:
        invalid += 1
        result_dict['judge'] = 'invalid'
    elif pred_result != INVALID_ANS and abs(float(gold_result) - float(pred_result)) < 1e-4:
        correct += 1
        result_dict['judge'] = 'true'
    else:
        result_dict['judge'] = 'false'
    writer.write(result_dict)
print("acc: ", correct/len(test_data))
print("invalid: ",invalid/len(test_data))
