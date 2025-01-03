# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)
from transformers import LlamaForCausalLM
import deepspeed

from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from sft_utils import Ffunc, init_wandb
import yaml

"""
accelerate launch sft_llama2.py \
    --output_dir="./sft" \
    --max_steps=1000 \
    --logging_steps=10 \
    --save_steps=10 \
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
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_llama2" \
    --report_to="wandb"
"""

import warnings


class myConstantLengthDataset(ConstantLengthDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        # if formatting_func is not None:
        #     if formatting_func.__code__.co_argcount > 1:
        #         warnings.warn(
        #             "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
        #             " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
        #         )


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="../ckpts/meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "the model name"},
    )
    dataset_name: Optional[str] = field(
        default="../hf_datasets/lvwerra/stack-exchange-paired",
        metadata={"help": "the dataset name"},
    )
    subset: Optional[str] = field(
        default="data/finetune", metadata={"help": "the subset to use"}
    )
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "the size of the validation set"}
    )
    streaming: Optional[bool] = field(
        default=True, metadata={"help": "whether to stream the dataset"}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "the shuffle buffer size"}
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "the sequence length"}
    )
    num_workers: Optional[int] = field(
        default=4, metadata={"help": "the number of workers"}
    )
    use_bnb: Optional[bool] = field(
        default=True, metadata={"help": "whether to use BitsAndBytes"}
    )
    template_str_dir: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, SFTConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
training_args.ddp_find_unused_parameters = False
training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and training_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)


def chars_token_ratio(dataset, tokenizer, nb_examples=400, prepare_sample_text=None):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# def prepare_sample_text(example):
#     """Prepare the text from a sample of the dataset."""
#     text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
#     return text


def create_datasets(tokenizer, args, seed=None, prepare_sample_text=None):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(
        train_data, tokenizer, prepare_sample_text=prepare_sample_text
    )
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = myConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = myConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
from accelerate import PartialState

device_string = PartialState().process_index
# model_init_kwargs = {
#     "device_map": {"": device_string},
#     "torch_dtype": torch.float16,
# }
base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": device_string},
    trust_remote_code=True,
    use_auth_token=True,
    torch_dtype=torch.float16,
)
base_model.config.use_cache = False
# init_wandb(script_args)

tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
with open(script_args.template_str_dir, "r") as f:
    cfg = yaml.safe_load(f)
prepare_sample_text = Ffunc(
    tokenizer=tokenizer,
    dataset_text_field="question",
    output_text_field="response_j",
    instruct_template=cfg["instruction_template"],
)

train_dataset, eval_dataset = create_datasets(
    tokenizer,
    script_args,
    seed=training_args.seed,
    prepare_sample_text=prepare_sample_text,
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    formatting_func=prepare_sample_text,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_torch_xpu_available():
    torch.xpu.empty_cache()
elif is_torch_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
