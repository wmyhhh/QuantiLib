import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from ..base import BaseQuantizer

class BnBQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        device_map="auto",
        quant_type="4bit",   # "4bit" 或 "8bit"
        save_tokenizer=True,
        save_dir=None,
        bnb_4bit_compute_dtype="float16",  # "float16" / "bfloat16"
        bnb_4bit_quant_type="nf4",         # "nf4" / "fp4"
        bnb_4bit_use_double_quant=False,   # 是否启用 double quant
        **kwargs
    ):
        super().__init__(
            model=model,
            device_map=device_map,
            quant_type=quant_type,
            save_tokenizer=save_tokenizer,
            save_dir=save_dir or "bnb",
            **kwargs
        )

        # 保存额外参数
        self.bnb_4bit_compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }.get(bnb_4bit_compute_dtype, torch.float16)

        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

    def quantize(self):
        if not isinstance(self.model_name_or_path, str):
            print("[Warning] 手动传入 nn.Module 目前不做实际量化，直接返回原模型")
            self.quantized = self.model
            return self.quantized

        if self.quant_type == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
            print(f"[BnB] 4bit 量化配置: {quant_config}")
        elif self.quant_type == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"[BnB] 8bit 量化配置: {quant_config}")
        else:
            raise ValueError(f"[BnB] Unsupported quant_type: {self.quant_type}")

        print(f"[BnB] 加载并量化模型 {self.model_name_or_path} ...")
        self.quantized = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quant_config,
            device_map=self.device_map,
        )
        #self.quantized = True
        return self.quantized


    def save(self, save_dir: str):
        if self.quantized is None:
            raise RuntimeError("请先调用 quantize() 再保存模型")
        print(f"Saving quantized model to {save_dir} ...")
        self.quantized.save_pretrained(save_dir)
