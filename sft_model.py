# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

""" 
This is from /hpc2hdd/home/sguo349/.conda/envs/whd/lib/python3.11/site-packages/trl/commands/scripts/sft.py
"""
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext
from transformers import HfArgumentParser
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from utils import Ffunc

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


@dataclass
class MySFTargs(SFTScriptArguments):
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"help": "the dataset name"},
    )
    content_field: str = field(
        default="text",
        metadata={"help": "the dataset name"},
    )
    response_field: str = field(
        default="answer",
        metadata={"help": "the dataset name"},
    )
    train_file_dir: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    eval_file_dir: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    template_str_dir: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    human_template: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    response_template: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )


if __name__ == "__main__":
    import sys
    import yaml

    config_file, new_output_dir = sys.argv[1], sys.argv[2]

    parser = HfArgumentParser((MySFTargs, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_json_file(json_file=config_file)

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    with open(args.template_str_dir, "r") as f:
        cfg = yaml.safe_load(f)
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    formatting_function = Ffunc(
        tokenizer,
        dataset_text_field=args.content_field,
        output_text_field=args.response_field,
        system_prompt=args.system_prompt,
        instruct_template=cfg["instruction_template"],
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        args.response_template,
        instruction_template=args.human_template,
        tokenizer=tokenizer,
    )
    ################
    # Dataset
    ################

    train_dataset = load_dataset("json", data_files=args.train_file_dir, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file_dir, split="train")
    ################
    # Optional rich context managers
    ###############
    training_args.output_dir = new_output_dir
    training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the SFTTrainer...")
    )

    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Training
    ################
    from accelerate import PartialState
    device_string = PartialState().process_index
    model_init_kwargs = {"device_map":{'':device_string},'torch_dtype': torch.float16}
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            formatting_func=formatting_function,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            model_init_kwargs=model_init_kwargs
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
