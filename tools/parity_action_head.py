"""Dump one action chunk with the fast-attention path on or off, for a value diff."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evo_depth")))
from scripts.Evo_depth_server import load_model_and_normalizer

CKPT = os.environ.get("EVO_DEPTH_CKPT_DIR", "/home/nvidia/evo_weights/lib/libero_spatial")
TAG = os.environ.get("PARITY_TAG", "0")


def main():
    model, normalizer = load_model_and_normalizer(CKPT)
    g = torch.Generator(device="cpu").manual_seed(7)
    images = [torch.rand((3, 448, 448), generator=g).to("cuda") for _ in range(3)]
    image_mask = torch.zeros((3,), dtype=torch.int32, device="cuda")
    state = torch.zeros((1, 24), dtype=torch.float32, device="cuda")

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        fused = model.get_vl_embeddings(images=images, image_mask=image_mask, prompt="dummy prompt")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.manual_seed(0)
        action = model.predict_action(fused, state, action_mask=torch.tensor(
            [[1] * 7 + [0] * 17], dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

    a = action.float().cpu().numpy()
    np.save(f"/home/nvidia/evo_depth_jetson/results/action_fast{TAG}.npy", a)
    print(json.dumps({"tag": TAG, "shape": list(a.shape), "action_ms": round(dt, 2)}))


main()
