# QuantiLib

## Introduction
QuantiLib 是一个集成的 CLI 工具，用于实现简便的大模型量化，目前支持以下方法：
- **BitsAndBytes (BNB)**
- **GPTQ**
- **AWQ**

| 方法       | 支持位数      | 特点               |
| -------- | --------- | ---------------- |
| **BNB**  | 4/8 bit   | 高效推理，支持 NF4/FP4  |
| **GPTQ** | 2/3/4 bit | 校准量化，适合精度要求较高的任务 |
| **AWQ**  | 2/3/4 bit | 激活感知量化，适合推理部署    |


未来将支持更多方法...

---

## 📦 Installation

```bash
git clone https://github.com/yourname/QuantiLib.git
cd QuantiLib
pip install -r requirements.txt
```

## Usage

### Example
```bash
# BNB 4-bit 量化
python3 -m QuantiLib.cli \
    --model_name "Qwen/Qwen3-1.7B" \
    --method bnb \
    --save_tokenizer False \
    --bnb_4bit_use_double_quant True \
    --save_dir ./bnb_4bit

# GPTQ 量化
python3 -m QuantiLib.cli \
    --model_name "Qwen/Qwen3-1.7B" \
    --method gptq \
    --batch_size 1 \
    --gptq_group_size 128
```
## Arguments

### Base arguments
| 参数                 | 说明                       |
| ------------------ | ------------------------ |
| `--model_name`     | HuggingFace 模型名          |
| `--model_path`     | 本地模型路径                   |
| `--method`         | 量化方法 {bnb,gptq,awq,aqlm} |
| `--device_map`     | {auto,cuda,cpu}          |
| `--save_dir`       | 模型保存路径                   |
| `--save_tokenizer` | {True,False} 是否保存分词器     |

### BnB arguments
| 参数                            | 说明                         |
| ----------------------------- | -------------------------- |
| `--quant_type`                | {4bit,8bit}                |
| `--bnb_4bit_compute_type`     | {float16,bfloat16,float32} |
| `--bnb_4bit_quant_type`       | {nf4,fp4}                  |
| `--bnb_4bit_use_double_quant` | {True,False}               |

### GPTQ arguments
| 参数                  | 说明                |
| ------------------- | ----------------- |
| `--quant_type`      | {2bit,3bit,4bit}  |
| `--batch_size`      | 校准时的 batch size   |
| `--calib_dataset`   | 校准数据集             |
| `--gptq_group_size` | GPTQ 的 group size |


### AWQ arguments
| 参数             | 说明               |
| -------------- | ---------------- |
| `--quant_type` | {2bit,3bit,4bit} |
| `--group_size` | AWQ 的 group size |
