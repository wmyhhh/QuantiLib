from .bnb import BnBQuantizer
from .gptq import GPTQQuantizer
from .aqlm import AQLMQuantizer
from .awq import AWQQuantizer

QUANTIZER_REGISTRY = {
    "bnb": BnBQuantizer,
    "gptq": GPTQQuantizer,
    "aqlm": AQLMQuantizer,
    "awq": AWQQuantizer
}

bnb_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['4bit', '8bit'],  # "4bit" or "8bit"
    "bnb_4bit_compute_dtype": ['float32', 'bfloat16'],  # Only for 4bit
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],
    "bnb_4bit_quant_type": ["nf4", "fp4"], 
    "bnb_4bit_use_double_quant": [True, False] # Only for 4bit
}

gptq_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['2bit', '3bit', '4bit'],  # "2bit", "3bit", or "4bit"
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],
    "batch_size": [int],  # 校准时的 batch size
    "calib_dataset": [str, list],  # 校准数据集
    "gptq_group_size": [int],  # GPTQ 的 group size
}

awq_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['2bit', '3bit', '4bit'],  # "2bit", "3bit", or "4bit"
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],  
    "group_size": [int],  # AWQ 的 group size
}

aqlm_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['8bit'],  # Currently only "8bit" is supported
    "dataset_path": [str],  # Path to the calibration dataset
    "nsamples": [int],  # Number of samples to use for calibration
    "val_size": [int],  # Validation set size
    "num_codebooks": [int],  # Number of codebooks
    "in_group_size": [int],
    "local_batch_size": [int],
    "offload_activations": [True, False],
    "save_dir": [str],
    "device_map": "cuda",
    "attn_implementation": "eager",
    "save_tokenizer": [True, False],
}

METHOD_ARGS = {
    "gptq": gptq_args,
    "bnb": bnb_args,
    "awq": awq_args,
    "aqlm": aqlm_args
}

