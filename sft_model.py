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
from data_set_to_load import *
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import warnings
from transformers import DataCollatorForLanguageModeling


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
from sft_utils import Ffunc, get_raw_dataset, get_max_samples, init_wandb

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
    sft_task_ds: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    data_home_dir: str = field(default=None, metadata={"help": "the dataset name"})
    response_field: str = field(
        default="answer",
        metadata={"help": "the dataset name"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    train_file: str = field(
        default=None,
        metadata={"help": "the dataset name"},
    )
    eval_file: str = field(
        default=None,
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


class myDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )

        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if (
            not self.mlm
            and self.instruction_template
            and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(
                    batch["labels"][i] == self.response_token_ids[0]
                )[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            assistant_idx : assistant_idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_idxs.append(
                            assistant_idx + len(self.response_token_ids)
                        )

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if (
                        human_token_ids
                        == batch["labels"][i][
                            human_idx : human_idx + len(human_token_ids)
                        ].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(
                    zip(human_token_ids_idxs, response_token_ids_idxs)
                ):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = (
                attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            )
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch


if __name__ == "__main__":
    import yaml
    import json
    import sys

    parser = HfArgumentParser((MySFTargs, SFTConfig, ModelConfig))
    config_file, output_file = sys.argv[1], sys.argv[2]
    run_name = config_file.split("/")[-1].split(".")[0]
    args, training_args, model_config = parser.parse_json_file(json_file=config_file)
    if args.sft_task_ds:
        args.system_prompt = SYS_PROMPT4DS[args.sft_task_ds]
    config_json: dict = json.load(open(config_file, "r"))
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    if args.template_str_dir is not None:
        with open(args.template_str_dir, "r") as f:
            cfg = yaml.safe_load(f)
    training_args.output_dir = output_file
    ################
    # Model init kwargs & Tokenizer
    ################
    init_wandb(config_json, name=run_name)
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
        instruct_template=(
            cfg["instruction_template"] if args.template_str_dir is not None else None
        ),
    )
    data_collator = myDataCollatorForCompletionOnlyLM(
        args.response_template,
        instruction_template=args.human_template,
        tokenizer=tokenizer,
    )
    ################
    # Dataset
    ################

    train_dataset, eval_dataset = get_raw_dataset(args)

    if args.max_train_samples is not None:
        train_dataset = get_max_samples(train_dataset, args.max_train_samples)
        eval_dataset = get_max_samples(eval_dataset, args.max_eval_samples)
        print(
            f"load train datasets first {args.max_train_samples}, eval dataset first {args.max_eval_samples}"
        )
    ################
    # Optional rich context managers
    ###############
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
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
    model_init_kwargs = {
        "device_map": {"": device_string},
        "torch_dtype": torch.float16,
    }

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
            model_init_kwargs=model_init_kwargs,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
