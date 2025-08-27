# llm_quant/base.py
from abc import ABC, abstractmethod
import os

class BaseQuantizer(ABC):
    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        quant_type: str = None,
        save_tokenizer: bool = True,
        save_dir: str = None,
        **kwargs
    ):
        self.model_name_or_path = model  # HuggingFace 模型名或本地路径
        self.device_map = device_map
        self.quant_type = quant_type
        self.save_tokenizer = save_tokenizer
        self.save_dir = save_dir or self.__class__.__name__.replace("Quantizer", "").lower()
        self.model = None
        self.quantized = False
        self.kwargs = kwargs

    @abstractmethod
    def quantize(self):
        """
        执行量化操作
        必须返回量化后的模型
        """
        pass

  
    def save(self, save_dir: str, save_tokenizer: bool = True):
        """
        通用保存逻辑：
        - 优先调用 Hugging Face 风格的 save_pretrained
        - 否则尝试调用第三方库常见的 save
        - 可选保存 tokenizer
        子类一般无需覆写；若底层库极其特殊，再覆写。
        """
        if not self.quantized:
            raise RuntimeError("请先调用 quantize() 再保存模型！")

        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
        elif hasattr(self.model, "save"):
            self.model.save(save_dir)
        else:
            raise NotImplementedError(
                f"{type(self.model).__name__} 既无 save_pretrained 也无 save，无法通用保存。"
            )

        # 保存 tokenizer（优先用自带的；否则用 quantizer 持有的）
        if save_tokenizer:
            tok = getattr(self, "tokenizer", None)
            if tok is None:
                tok = getattr(self.model, "tokenizer", None)
            if tok is not None and hasattr(tok, "save_pretrained"):
                tok.save_pretrained(save_dir)
