DEFAULT_PAD_TOKEN = {"llama2": "[PAD]", "qwen2.5": "<|im_end|>"}
DEFAULT_EOS_TOKEN = {"llama2": "</s>", "qwen2.5": "<|im_end|>"}
DEFAULT_BOS_TOKEN = {"llama2": "<s>", "qwen2.5": "<|im_start|>"}
DEFAULT_UNK_TOKEN = "<unk>"
IGNORE_INDEX = -100
PROMPT_DICT = {
    "llama2": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "qwen2.5": (
        "<|im_start|>system\nBelow is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n<|im_end|>\n"
        "<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}

# from /opt/miniconda3/envs/whd/lib/python3.10/site-packages/peft/utils/constants.py
TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING = {
    "llama": ["input_layernorm", "post_attention_layernorm", "norm"],
    "bloom": ["input_layernorm", "post_attention_layernorm", "ln_f"],
    "llava": [
        "multi_modal_projector",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
        "embed_tokens",
        "lm_head",
    ],
    "t5": ["layer_norm", "final_layer_norm"],
    "mt5": ["layer_norm", "final_layer_norm"],
    "bart": ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"],
    "gpt2": ["ln_1", "ln_2", "ln_f"],
    "blip-2": ["layernorm", "LayerNorm", "final_layer_norm", "self_attn_layer_norm"],
    "gptj": ["ln_1", "ln_f"],
    "falcon": ["input_layernorm", "post_attention_layernorm", "ln_f"],
    "mistral": ["input_layernorm", "post_attention_layernorm", "norm"],
    "phi": ["input_layernorm", "final_layernorm"],
    "gemma": ["input_layernorm", "post_attention_layernorm", "norm"],
    "qwen2": ["post_attention_layernorm"],
}

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "t5": ["k", "v", "wo"],
    "mt5": ["k", "v", "wi_1"],
    "gpt2": ["c_attn", "mlp.c_proj"],
    "bloom": ["query_key_value", "mlp.dense_4h_to_h"],
    "roberta": ["key", "value", "output.dense"],
    "opt": ["q_proj", "k_proj", "fc2"],
    "gptj": ["q_proj", "v_proj", "fc_out"],
    "gpt_neox": ["query_key_value", "dense_4h_to_h"],
    "gpt_neo": ["q_proj", "v_proj", "c_proj"],
    "bart": ["q_proj", "v_proj", "fc2"],
    "gpt_bigcode": ["c_attn", "mlp.c_proj"],
    "llama": ["k_proj", "v_proj", "down_proj"],
    "mistral": ["k_proj", "v_proj", "down_proj"],
    "mixtral": ["k_proj", "v_proj", "w2"],
    "bert": ["key", "value", "output.dense"],
    "deberta-v2": ["key_proj", "value_proj", "output.dense"],
    "deberta": ["in_proj", "output.dense"],
    "RefinedWebModel": ["query_key_value", "dense_4h_to_h"],
    "RefinedWeb": ["query_key_value", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense_4h_to_h"],
    "phi": ["q_proj", "v_proj", "fc2"],
    "gemma": ["q_proj", "v_proj", "down_proj"],
    "qwen2": ["q_proj", "v_proj", "down_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {
    "t5": ["wo"],
    "mt5": [],
    "gpt2": ["mlp.c_proj"],
    "bloom": ["mlp.dense_4h_to_h"],
    "roberta": ["output.dense"],
    "opt": ["fc2"],
    "gptj": ["fc_out"],
    "gpt_neox": ["dense_4h_to_h"],
    "gpt_neo": ["c_proj"],
    "bart": ["fc2"],
    "gpt_bigcode": ["mlp.c_proj"],
    "llama": ["down_proj"],
    "mistral": ["down_proj"],
    "mixtral": ["w2"],
    "bert": ["output.dense"],
    "deberta-v2": ["output.dense"],
    "deberta": ["output.dense"],
    "RefinedWeb": ["dense_4h_to_h"],
    "RefinedWebModel": ["dense_4h_to_h"],
    "falcon": ["dense_4h_to_h"],
    "phi": ["fc2"],
    "gemma": ["down_proj"],
    "qwen2": ["down_proj"],
}

TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "gpt_bigcode": ["c_attn"],
    "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
    "qwen2": ["q_proj", "v_proj"],
}

TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    # "btlm": ["c_proj", "c_attn"],  # tested, does not work because of different shapes
    "codegen": ["qkv_proj"],
    # "mistral": ["q_proj", "v_proj"],  # tested, does not work because of different shapes
    # "mixtral": ["q_proj", "v_proj"],  # tested, does not work because of different shapes
    "stablelm": ["q_proj", "v_proj"],
    # "phi": ["q_proj", "v_proj", "fc1", "fc2"],  # tested, does not work because of different shapes
    "phi": ["q_proj", "v_proj"],
    # "gemma": ["q_proj", "v_proj"],  # tested, does not work because of different shapes
    "qwen2": ["q_proj", "v_proj"],
}

TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["mlp.c_proj"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["mlp.c_proj"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
}

WEIGHTS_NAME = "adapter_model.bin"
SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]
INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
DUMMY_TARGET_MODULES = "dummy-target-modules"
