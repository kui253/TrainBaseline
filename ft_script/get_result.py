import re
import jsonlines
from tqdm import tqdm
import sys

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
test_data_dir, result_output_dir = sys.argv[1:]
with jsonlines.open(test_data_dir, 'r') as f:
    test_data = [obj for obj in f]




    # from peft import PeftModel
    # model = PeftModel.from_pretrained(model, "output_ckpt/lora_ver2")
invalid = 0
correct = 0
times = 0
with jsonlines.open(result_output_dir, 'r') as f:
    pred_data = [obj for obj in f]
for item_g, item_p in tqdm(zip(test_data, pred_data), desc="generating"):
    
    

    # 解码生成的文本
    gold_result = extract_answer(item_g["target"][0])
    pred_result = extract_answer(item_p["predict"])
    if pred_result == INVALID_ANS:
        invalid += 1
    elif pred_result != INVALID_ANS and abs(float(gold_result) - float(pred_result)) < 1e-4:
        correct += 1
    else:
        print("***********************************************************************")
        print("***********************************************************************")
        print("Gold: ",item_g["target"][0])
        print("Pred: ", item_p["predict"])
        times += 1
        if times > 5:
            
            break
        pass
print("acc: ", correct/len(test_data))
print("invalid: ",invalid/len(test_data))
