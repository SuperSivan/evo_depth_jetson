"""Export DA3's DPT network (cross-view, 1x3x3x504x504 -> 1x3x1296x1536) to ONNX.

Tries fp16 on the GPU first; the joint attention over 3888 tokens is the memory
pressure point, so it falls back to a CPU export when the GPU cannot hold it.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evo_depth")))
from scripts.Evo_depth_server import load_model_and_normalizer

CKPT = "/home/nvidia/evo_weights/lib/libero_spatial"
ROOT = "/home/nvidia/evo_depth_jetson/export"
DTYPE = os.environ.get("EXPORT_DTYPE", "fp16")
OUT = os.path.join(ROOT, f"da3_dpt_{DTYPE}.onnx")

class Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, imgs):
        return self.net._get_dpt_embedding(imgs)


def _exportable_cartesian_prod(*tensors):
    """torch.cartesian_prod has no ONNX symbolic. For 1-D inputs meshgrid+stack with
    indexing='ij' gives the identical result, and both pieces export."""
    grids = torch.meshgrid(*tensors, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def main():
    torch.cartesian_prod = _exportable_cartesian_prod
    os.makedirs(ROOT, exist_ok=True)
    model, _ = load_model_and_normalizer(CKPT)
    api = model.embedder.dpt_embedder.model
    net = api.model

    torch.manual_seed(0)
    dummy = torch.rand(1, 3, 3, 504, 504, device="cuda")
    with torch.no_grad():
        ref = net._get_dpt_embedding(dummy)
    print("reference:", tuple(ref.shape), ref.dtype)
    np.save(os.path.join(ROOT, "da3_ref.npy"), ref.detach().float().cpu().numpy())
    np.save(os.path.join(ROOT, "da3_in.npy"), dummy.detach().float().cpu().numpy())

    wrap = Wrap(net)
    if DTYPE == "fp16":
        wrap = wrap.half()
        dummy = dummy.half()
    else:
        wrap = wrap.float()
        dummy = dummy.float()
    wrap = wrap.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrap, (dummy,), OUT,
            input_names=["imgs"], output_names=["feats"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print("onnx written:", OUT, os.path.getsize(OUT) / 1e6, "MB  dtype:", DTYPE)
    print("DA3_EXPORT_DONE")


main()
