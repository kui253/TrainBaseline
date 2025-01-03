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

import torch
import transformers
from mymodel_llama2 import LlamaSNRFT
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig
import jsonlines
import re

from system_prompt_template import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lambda_loss: Optional[float] = field(default=0.001)
    if_lora: Optional[bool] = field(default=False)
    if_embed_norm: Optional[bool] = field(default=False)
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
    response_field: str = field(default="target", metadata={"help": "model template"})
    reject_field: str = field(default="reject", metadata={"help": "model template"})
    chosen_field: str = field(default="accept", metadata={"help": "model template"})
    num_proc: int = field(default=16, metadata={"help": "nrft mode to choose"})
    per_proc_batch: int = field(default=100, metadata={"help": "nrft mode to choose"})
    ft_type: str = field(default="sft", metadata={"help": "nrft mode to choose"})
    ## TODO
    rationale_max_len: int = field(default=64)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
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


# def _tokenize_fn(
#     strings: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
#     max_length: int = None,
# ) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length if max_length is None else max_length,
#             truncation=True,
#         )
#         for text in strings
#     ]
#     input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
#     input_ids_lens = labels_lens = [
#         tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
#         for tokenized in tokenized_list
#     ]
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )


# def preprocess(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [
#         _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
#     ]
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
#     return dict(input_ids=input_ids, labels=labels)


# TODO
"""
1. 可能需要有 题设+ 推理，每次推理都是从题设开始，逐渐增多，这个时候不用加入sys prompt，之后再sft, 需要shift
2. 去除题设，每次推理都是从一个固定的sentence lens开始，1，2
3. 加入题设，但是每次推理是滑动的。
"""


# def preprocess_next_rationale(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
#     model_template: str = "llama2",
#     rationale_max_len: int = 64,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     sources_tokenized, targets_tokenized = [
#         _tokenize_fn(strings, tokenizer) for strings in (sources, targets)
#     ]
#     input_ids = sources_tokenized["input_ids"]
#     labels = targets_tokenized["input_ids"]
#     padded_labels = []
#     padded_input_ids = []
#     for input_id, label in zip(input_ids, labels):

#         if len(input_id) > len(label):
#             padding = torch.full(
#                 (abs(len(input_id) - len(label)),), IGNORE_INDEX, dtype=label.dtype
#             )
#             label = torch.cat((label, padding), dim=0)

#         else:

#             padding = torch.full(
#                 (abs(len(input_id) - len(label)),),
#                 tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN[model_template]),
#                 dtype=label.dtype,
#             )
#             input_id = torch.cat((input_id, padding), dim=0)
#         padded_input_ids.append(input_id)
#         padded_labels.append(label)

#     return dict(input_ids=padded_input_ids, labels=padded_labels)


# class NextRDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(
#         self,
#         data_path: str,
#         tokenizer: transformers.PreTrainedTokenizer,
#         ft_mode="sft",
#         mode=1,
#         model_template: str = "llama2",
#         rationale_max_len: int = 64,
#     ):
#         super(NextRDataset, self).__init__()
#         logging.warning("Loading data...")
#         with jsonlines.open(data_path, "r") as f:

#             list_data_dict = [obj for obj in f]
#         logging.warning("Formatting inputs...")
#         sources = []
#         targets = []
#         self.ft_mode = ft_mode
#         if ft_mode == "nrft":
#             for item in tqdm(list_data_dict, desc="processing"):
#                 s_str: str = item["source"][0]
#                 t_str: str = item["target"][0]
#                 cache_s_str = s_str

#                 steps = t_str.split("\n")

#                 if mode == 1:
#                     for step_id in range(len(steps)):
#                         sources.append(cache_s_str)
#                         targets.append(steps[step_id])
#                         cache_s_str += " " + steps[step_id]
#                 elif mode == 2:
#                     for step_id in range(len(steps) - 1):
#                         sources.append(steps[step_id])
#                         targets.append(steps[step_id + 1])
#                 else:
#                     for step_id in range(len(steps) - 1):
#                         sources.append(s_str + " " + (steps[step_id]))
#                         targets.append(steps[step_id + 1])
#             data_dict = preprocess_next_rationale(
#                 sources,
#                 targets,
#                 tokenizer,
#                 model_template=model_template,
#                 rationale_max_len=rationale_max_len,
#             )
#         elif ft_mode == "sft":
#             prompt_no_input = PROMPT_DICT[model_template]
#             sources = [
#                 prompt_no_input.format_map({"instruction": example["source"][0]})
#                 for example in list_data_dict
#             ]
#             targets = [
#                 example["target"][0] + f"{tokenizer.eos_token}"
#                 for example in list_data_dict
#             ]

#             logging.warning("Tokenizing inputs... This may take some time...")
#             data_dict = preprocess(sources, targets, tokenizer)
#         elif ft_mode == "snrft":
#             prompt_no_input = PROMPT_DICT[model_template]
#             sources = [
#                 prompt_no_input.format_map({"instruction": example["source"][0]})
#                 for example in list_data_dict
#             ]
#             targets = [
#                 example["target"][0] + f"{tokenizer.eos_token}"
#                 for example in list_data_dict
#             ]
#             steps = []
#             all_length = []
#             for item in list_data_dict:
#                 steps_temp = item["target"][0].split("\n")
#                 all_length.append(len(steps_temp))
#                 steps.append(steps_temp)
#             logging.warning("Tokenizing inputs... This may take some time...")
#             data_dict = preprocess(sources, targets, tokenizer)
#             logging.warning("Tokenizing sentences... This may take some time...")
#             tokenized_steps = []
#             for sample_steps in steps:
#                 data_dict_steps = tokenizer(
#                     sample_steps,
#                     return_tensors="pt",
#                     padding="max_length",
#                     max_length=rationale_max_len,
#                     truncation=True,
#                 )  # len(shape) = 2
#                 tokenized_steps.append(data_dict_steps["input_ids"])
#             self.lengths = all_length
#             self.steps = tokenized_steps

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         if self.ft_mode == "snrft":
#             return dict(
#                 input_ids=self.input_ids[i],
#                 labels=self.labels[i],
#                 batch_length=self.lengths[i],
#                 batch_steps=self.steps[i],
#             )
#         else:
#             return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         with jsonlines.open(data_path, "r") as f:

#             list_data_dict = [obj for obj in f]
#         logging.warning("Formatting inputs...")
#         prompt_no_input = PROMPT_DICT["llama2"]
#         sources = [
#             prompt_no_input.format_map({"instruction": example["source"][0]})
#             for example in list_data_dict
#         ]
#         targets = [
#             example["target"][0] + f"{tokenizer.eos_token}"
#             for example in list_data_dict
#         ]

#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# @dataclass
# class DataCollatorForSupervisedDataset_snrft(object):
#     """Collate examples for supervised fine-tuning."""

#     padding_steps: bool
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         if not instances[0].get("batch_length"):
#             input_ids, labels = tuple(
#                 [instance[key] for instance in instances]
#                 for key in ("input_ids", "labels")
#             )
#             input_ids = torch.nn.utils.rnn.pad_sequence(
#                 input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#             )
#             labels = torch.nn.utils.rnn.pad_sequence(
#                 labels, batch_first=True, padding_value=IGNORE_INDEX
#             )
#             return dict(
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#             )
#         else:

#             input_ids, labels, batch_length, batch_steps = tuple(
#                 [instance[key] for instance in instances]
#                 for key in ("input_ids", "labels", "batch_length", "batch_steps")
#             )
#             input_ids = torch.nn.utils.rnn.pad_sequence(
#                 input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#             )
#             labels = torch.nn.utils.rnn.pad_sequence(
#                 labels, batch_first=True, padding_value=IGNORE_INDEX
#             )
#             max_batch_length = max(batch_length)
#             rationale_seq_len = batch_steps[0].shape[1]
#             all_steps = []
#             all_steps_labels = []
#             for item in batch_steps:
#                 cur_len = item.shape[0]
#                 # padding
#                 if self.padding_steps:
#                     if cur_len == max_batch_length:
#                         padded_step = item

#                     else:

#                         padding = torch.full(
#                             (max_batch_length - cur_len, rationale_seq_len),
#                             self.tokenizer.pad_token_id,
#                             dtype=labels.dtype,
#                         )
#                         padded_step = torch.cat((item, padding), dim=0)
#                     step_label = padded_step[1:, :]
#                     step_input = padded_step[:-1, :]
#                 # without padding
#                 else:
#                     step_label = item[1:, :]
#                     step_input = item[:-1, :]
#                 all_steps.append(step_input)
#                 all_steps_labels.append(step_label)
#             padded_batch_step = torch.cat(all_steps, dim=0)
#             padded_batch_step_labels = torch.cat(all_steps_labels, dim=0)

#             return dict(
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#                 rationale_steps_input_ids=padded_batch_step,
#                 rationale_steps_attention_mask=padded_batch_step.ne(
#                     self.tokenizer.pad_token_id
#                 ),
#                 rationale_steps_labels=padded_batch_step_labels,
#             )
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

        tar_module = [
            "k_proj",
            "v_proj",
            "down_proj",
        ]
        format_model_name = re.sub(r"\d+$", "", data_args.model_template)
        param_name_space = (
            EMBEDDING_LAYER_NAMES
            + TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING[format_model_name]
        )
        peft_config = LoraConfig(
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dp,
            target_modules=tar_module,
            # modules_to_save=param_name_space if model_args.if_embed_norm else None,
        )
        model = get_peft_model(model, peft_config)
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("Trainable parameters:", trainable_params)
        # if model_args.if_embed_norm:
        #     print("activate embedding and layernorm")

        #     for name, param in model.named_parameters():
        #         if any(sub_name in name for sub_name in param_name_space):

        #             param.requires_grad = True

        #     # 验证可训练参数
        #     trainable_params = sum(
        #         p.numel() for p in model.parameters() if p.requires_grad
        #     )
        #     print("Trainable parameters:", trainable_params)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
