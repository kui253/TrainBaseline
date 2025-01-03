from datasets import load_dataset
import transformers
from trl.trainer.utils import DPODataCollatorWithPadding
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from system_prompt_template import IGNORE_INDEX
from process_fn import PROCESSFN


@dataclass
class SFTDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple(
            [torch.tensor(instance[key]) for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


COLLATOR = {"sft": SFTDataCollator, "dpo": None}


def from_json_file(json_dir, tokenizer, args):
    ds = load_dataset("json", data_files=json_dir, split="train")
    ds = ds.map(
        PROCESSFN[args.ft_type],
        num_proc=args.num_proc,
        batch_size=args.per_proc_batch,
        batched=True,
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
    )
    if COLLATOR[args.ft_type] is not None:
        cl = COLLATOR[args.ft_type](tokenizer)
    else:
        cl = None
    return ds, cl


def get_hf_dataset_and_collator(tokenizer, args):
    if args.data_path.endswith(".json"):
        return from_json_file(args.data_path, tokenizer, args)
    else:
        pass


# if __name__ == "__main__":
#     pass
