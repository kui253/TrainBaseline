from typing import Any
from transformers import AutoTokenizer
from datasets import load_dataset
from data_set_to_load import *


class Ffunc:
    def __init__(
        self,
        tokenizer,
        dataset_text_field,
        output_text_field,
        instruct_template: str,
        system_prompt: str = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_text_field = dataset_text_field
        self.output_text_field = output_text_field
        self.system_prompt = system_prompt
        self.ins_temp = instruct_template

    def __call__(self, input_examples) -> Any:

        if isinstance(self.dataset_text_field, list):
            input_part = []
            for values in zip(
                *(input_examples[key] for key in self.dataset_text_field)
            ):
                concat_str = []
                for v in values:

                    if isinstance(v, list):
                        v = "\n".join(v)
                    concat_str.append(v)
                concat_str = "\n".join(concat_str)
                input_part.append(concat_str)

        else:
            input_part = input_examples[self.dataset_text_field]
        if isinstance(self.output_text_field, list):

            output_part = []
            for values in zip(*(input_examples[key] for key in self.output_text_field)):
                concat_str = []
                for v in values:

                    if isinstance(v, list):
                        v = "\n".join(v)
                    concat_str.append(v)
                concat_str = "\n".join(concat_str)

                output_part.append(concat_str)
        else:
            output_part = input_examples[self.output_text_field]
        messages = [
            [
                {"role": "user", "content": i},
                {"role": "assistant", "content": j},
            ]
            for i, j in zip(
                input_part,
                output_part,
            )
        ]
        processed_data = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            chat_template=self.ins_temp,
            # add_generation_prompt=True
        )

        return processed_data


# datas = load_dataset(DIR2NAME["specialty"]["pubmedqa"])


def get_raw_dataset(args):
    if args.sft_task_ds is not None:
        raw_dataset = load_dataset(DIR2NAME[args.sft_task_ds])
        return raw_dataset[args.train_file_dir], raw_dataset[args.eval_file_dir]

    if args.data_home_dir is not None:
        if args.train_file is not None:
            raw_dataset = load_dataset(args.data_home_dir)
            return raw_dataset[args.train_file], raw_dataset[args.eval_file]
        elif args.train_file_dir is not None:
            train_dataset = load_dataset(
                args.data_home_dir + args.train_file_dir, split="train"
            )
            eval_dataset = load_dataset(
                args.data_home_dir + args.eval_file_dir, split="train"
            )
            return train_dataset, eval_dataset
        else:
            raise "both train_file and train_file_dir are none"
    else:
        if "." in args.train_file_dir:
            format_file = args.train_file_dir.split(".")[-1]
            if format_file == "json":
                train_dataset = load_dataset(
                    "json", data_files=args.train_file_dir, split="train"
                )
                if args.eval_file_dir:
                    eval_dataset = load_dataset(
                        "json", data_files=args.eval_file_dir, split="train"
                    )
                else:
                    eval_dataset = None
            else:
                raise f"not support {format_file}"
            return train_dataset, eval_dataset
        else:
            raise "not a file, maybe a dir"


def get_max_samples(datasets, max_samples):
    return datasets.select(range(max_samples))


def init_wandb(args, name):
    import wandb

    wandb.init(
        name=name,
        project="ft_LLM",
        # track hyperparameters and run metadata
        config=args,
    )
