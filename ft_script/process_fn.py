import transformers
import copy
from system_prompt_template import IGNORE_INDEX, PROMPT_DICT


def sft_preprocess(examples, tokenizer, args):
    # apply template
    source_formatted = []
    examples_merged = []
    for s, t in zip(examples[args.content_field], examples[args.response_field]):
        if isinstance(s, list):
            s = s[0]
        if isinstance(t, list):
            t = t[0]
        s_f = PROMPT_DICT[args.model_template].format_map(dict(instruction=s))
        source_formatted.append(s_f)
        examples_merged.append(s_f + t + tokenizer.eos_token)

    examples_tokenized = tokenizer(
        examples_merged,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    sources_tokenized = tokenizer(
        source_formatted,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    sources_tokenized_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in sources_tokenized["input_ids"]
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized_lens):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def dpo_preprocess(examples, tokenizer, args):
    # the tokenizer part is integrated in DPOTrainer
    source_formatted = []
    w_merged = []
    l_merged = []
    for s, w, l in zip(
        examples[args.content_field],
        examples[args.chosen_field],
        examples[args.reject_field],
    ):
        s_f = PROMPT_DICT[args.model_template].format_map(dict(instruction=s))
        source_formatted.append(s_f)
        w_merged.append(s_f + w + tokenizer.eos_token)
        l_merged.append(s_f + l + tokenizer.eos_token)

    return dict(prompt=source_formatted, chosen=w_merged, rejected=l_merged)


PROCESSFN = {"sft": sft_preprocess, "dpo": dpo_preprocess}
