import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer    
from ..base import BaseQuantizer

class AQLMQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        nsamples=1024,
        val_size=128,
        num_codebooks=1,
        in_group_size=8,
        local_batch_size=1,
        offload_activations=False,
        save_dir="aqlm",
        save_tokenizer=True,
        device_map="cuda",
        **kwargs
    ):
        super().__init__(
            model=model,
            save_dir=save_dir,
            save_tokenizer=save_tokenizer,
            device_map=device_map,
            **kwargs
        )
        self.nsamples = nsamples
        self.val_size = val_size
        self.num_codebooks = num_codebooks
        self.in_group_size = in_group_size
        self.local_batch_size = local_batch_size
        self.offload_activations = offload_activations

    def quantize(self):
        # 调用 AQLM 脚本
        cmd = [
            "python", "main.py",
            self.model,
            f"--nsamples={self.nsamples}",
            f"--val_size={self.val_size}",
            f"--num_codebooks={self.num_codebooks}",
            f"--in_group_size={self.in_group_size}",
            f"--local_batch_size={self.local_batch_size}",
            "--save", self.save_dir,
        ]
        if self.offload_activations:
            cmd.append("--offload_activations")

        print(f"[AQLM] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # 量化完成后加载模型
        print(f"[AQLM] 加载量化模型: {self.save_dir}")
        self.quantized = AutoModelForCausalLM.from_pretrained(
            self.save_dir,
            device_map=self.device_map,
        )
        return self.quantized

    def save(self, save_dir: str):
        if self.quantized is None:
            raise RuntimeError("请先调用 quantize() 再保存模型")
        print(f"[AQLM] 模型已保存在 {save_dir}（由 main.py 生成）")
