from typing import Any
from transformers import AutoTokenizer


class Ffunc:
    def __init__(
        self,
        tokenizer,
        dataset_text_field,
        output_text_field,
        system_prompt: str,
        instruct_template: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_text_field = dataset_text_field
        self.output_text_field = output_text_field
        self.system_prompt = system_prompt
        self.ins_temp = instruct_template

    def __call__(self, input_examples) -> Any:

        messages = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": i},
                {"role": "assistant", "content": j},
            ]
            for i, j in zip(
                input_examples[self.dataset_text_field],
                input_examples[self.output_text_field],
            )
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            chat_template=self.ins_temp,
            # add_generation_prompt=True
        )


