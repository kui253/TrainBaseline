#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from trl import DPOTrainer
import torch
import transformers
from mymodel_llama2 import LlamaSNRFT
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig
import jsonlines
from trl import DPOConfig


from system_prompt_template import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lambda_loss: Optional[float] = field(default=0.001)
    if_lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dp: Optional[float] = field(default=0.1)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    ft_mode: str = field(default="sft", metadata={"help": "mode to sft"})
    mode: int = field(default=1, metadata={"help": "nrft mode to choose"})
    model_template: str = field(default="llama2", metadata={"help": "model template"})
    step_padding: bool = field(default=False, metadata={"help": "model template"})
    content_field: str = field(default="source", metadata={"help": "model template"})
    reject_field: str = field(default="reject", metadata={"help": "model template"})
    chosen_field: str = field(default="accept", metadata={"help": "model template"})
    num_proc: int = field(default=1, metadata={"help": "nrft mode to choose"})
    ft_type: str = field(default="sft", metadata={"help": "nrft mode to choose"})
    per_proc_batch: int = field(default=10, metadata={"help": "nrft mode to choose"})
    ## TODO
    rationale_max_len: int = field(default=64)


@dataclass
class TrainingArguments(DPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


from data_utils import get_hf_dataset_and_collator


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset, data_collator = get_hf_dataset_and_collator(tokenizer, data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


from transformers import AutoModelForCausalLM


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.lambda_loss = model_args.lambda_loss
    if data_args.ft_mode == "snrft":
        model = LlamaSNRFT.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN[data_args.model_template]
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN[data_args.model_template]
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN[data_args.model_template]
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    if model_args.if_lora:
        from peft import get_peft_config, get_peft_model, LoraConfig

        peft_config = LoraConfig(
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dp,
            target_modules="all-linear",
        )
        if data_args.ft_type != "dpo":

            model = get_peft_model(model, peft_config)
    else:
        peft_config = None
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        peft_config=peft_config
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
