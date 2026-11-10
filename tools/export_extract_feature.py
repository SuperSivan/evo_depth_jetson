"""Export InternVL3's extract_feature (ViT + pixel_shuffle + mlp1) to ONNX for TensorRT.

Exported in fp32; fp16 is chosen by trtexec so accumulation stays fp32.
"""
import os, sys, torch
from transformers import AutoModel

sys.path.insert(0, "/home/nvidia/evo_depth_jetson/Evo_depth")
VLM = "/home/nvidia/evo_depth_jetson/weights/InternVL3-1B"
OUT = os.environ.get("ONNX_OUT", "/home/nvidia/evo_depth_jetson/export/internvit_extract_feature.onnx")
BATCH = int(os.environ.get("BATCH", "1"))


class Wrap(torch.nn.Module):
    def __init__(self, chat_model):
        super().__init__()
        self.chat = chat_model

    def forward(self, pixel_values):
        return self.chat.extract_feature(pixel_values)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m = AutoModel.from_pretrained(VLM, trust_remote_code=True, torch_dtype=torch.float32,
                                  use_flash_attn=False, low_cpu_mem_usage=True, _fast_init=False)
    m = m.float().cuda().eval()
    wrap = Wrap(m).eval()
    dummy = torch.randn(BATCH, 3, 448, 448, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        ref = wrap(dummy)
    print("reference out:", tuple(ref.shape), ref.dtype)

    class Out(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = w

        def forward(self, pixel_values):
            return self.w(pixel_values)

    with torch.no_grad():
        torch.onnx.export(
            Out(wrap), (dummy,), OUT,
            input_names=["pixel_values"], output_names=["vit_embeds"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print("onnx written:", OUT, os.path.getsize(OUT) / 1e6, "MB")
    import numpy as np
    np.save("/home/nvidia/evo_depth_jetson/export/extract_feature_ref.npy",
            ref.detach().float().cpu().numpy())
    np.save("/home/nvidia/evo_depth_jetson/export/extract_feature_in.npy",
            dummy.detach().float().cpu().numpy())
    print("EXPORT_DONE")


main()
