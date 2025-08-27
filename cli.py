import argparse
import os
import shutil
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

from .methods import *  # 确保量化方法被注册
from .methods.args_registry import METHOD_ARGS, QUANTIZER_REGISTRY


# copy tokenizer files
def copy_tokenizer(src, dst):
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    for f in tokenizer_files:
        src_file = os.path.join(src, f)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst)
            print(f"[Info] 已复制 {f} 到 {dst}")


def validate_args(method, args_dict):
    """检查参数是否合法"""
    if method not in METHOD_ARGS:
        raise ValueError(f"未知的量化方法: {method}, 可选: {list(METHOD_ARGS.keys())}")
    
    valid_args = METHOD_ARGS[method]    # returns a dict(e.g. bnb_args)

    # k: keys(e.g. quanti_type), v: values(e.g. ['4bit', '8bit'])
    for k, v in args_dict.items():
        # 检查参数名是否有效
        if k not in valid_args:
            raise ValueError(f"参数 {k} 不属于 {method} 的参数集合: {list(valid_args.keys())}")
        
        allowed_types_or_vals = valid_args[k]

        # 判断值是否符合
        if isinstance(allowed_types_or_vals[0], type):
            # 类型约束
            if not isinstance(v, tuple(allowed_types_or_vals)):
                raise TypeError(f"参数 {k} 需要类型 {allowed_types_or_vals}, 但传入 {type(v)}")
        else:
            # 值枚举约束
            if v not in allowed_types_or_vals:
                raise ValueError(f"参数 {k} 的值 {v} 不在允许范围 {allowed_types_or_vals}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("--model_name", type=str, help="HuggingFace 模型名，例如 gpt2")
    parser.add_argument("--model_path", type=str, help="本地模型路径")
    parser.add_argument("--method", type=str, required=True, help="选择量化方法: gptq, bnb 等")
    parser.add_argument("--quant_type", type=str)
    parser.add_argument("--device_map", type=str)
    parser.add_argument("--save_tokenizer", type=bool)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str)
    parser.add_argument("--bnb_4bit_quant_type", type=str)
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--calib_dataset", type=str)
    parser.add_argument("--gptq_group_size", type=int)
    parser.add_argument("--gptq_act_order", type=bool)

    args = parser.parse_args()
    args_dict = vars(args)  # Convert Namespace to dict

    method = args_dict.pop("method")    # get the method and remove it from args_dict

    # 校验参数
    try:
        validate_args(method, {k: v for k, v in args_dict.items() if v is not None})
    except (ValueError, TypeError) as e:
        print(f"[参数错误] {e}")
        sys.exit(1)

    # 找到对应 Quantizer 并调用
    if method not in QUANTIZER_REGISTRY:
        print(f"未注册的量化方法: {method}")
        sys.exit(1)

    if args.model_path:
        model_name_or_path = args.model_path
    elif args.model_name:
        model_name_or_path = args.model_name
    else:
        raise ValueError("请提供 --model_name 或 --model_path 中至少一个")
    

    QuantizerClass = QUANTIZER_REGISTRY[method]
    model= model_name_or_path
    quantizer = QuantizerClass(model, **{k: v for k, v in args_dict.items() if v is not None})
    quantizer.quantize()
    print(f"✅ 成功完成 {method} 量化")


    # 保存路径使用量化器内部的 save_dir 默认值
    save_path = os.path.abspath(quantizer.save_dir)
    quantizer.save(save_path)
    print(f"量化模型已保存到: {save_path}")


    # 可选：复制 tokenizer
    if args.save_tokenizer:
        copy_tokenizer(model_name_or_path, save_path)
        print("[Info] tokenizer 文件已保存到量化目录")


if __name__ == "__main__":
    main()
