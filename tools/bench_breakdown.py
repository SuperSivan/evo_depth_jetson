"""Per-stage latency for Evo-Depth on the board. Mirrors benchmark_evo_depth.py's
inputs so the totals stay comparable, then times the three stages separately."""
import json, sys, os, time, statistics
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evo_depth")))
from scripts.Evo_depth_server import load_model_and_normalizer

CKPT = os.environ.get("EVO_DEPTH_CKPT_DIR", "/home/nvidia/evo_weights/lib/libero_spatial")
WARMUP = int(os.environ.get("WARMUP", "3"))
ITERS = int(os.environ.get("ITERS", "10"))


def build_inputs(device="cuda", mask_len=1):
    g = torch.Generator(device="cpu").manual_seed(1234)
    images = [torch.rand((3, 448, 448), generator=g).to(device) for _ in range(3)]
    state = torch.zeros((1, 24), dtype=torch.float32, device=device)
    image_mask = torch.zeros((mask_len,), dtype=torch.int32, device=device)
    action_mask = torch.tensor([[1] * 7 + [0] * 17], dtype=torch.int32, device=device)
    return images, state, image_mask, action_mask, "dummy prompt"


def timeit(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(statistics.mean(xs), 2),
            "p50": round(statistics.median(xs), 2), "min": round(xs[0], 2),
            "max": round(xs[-1], 2)}


def run(mask_len, label):
    model, normalizer = load_model_and_normalizer(CKPT)
    images, state, image_mask, action_mask, prompt = build_inputs(mask_len=mask_len)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(WARMUP):
            model.run_inference(images=images, image_mask=image_mask, prompt=prompt,
                                state_input=norm_state, action_mask=action_mask)

        vl, st, ac, tot = [], [], [], []
        for _ in range(ITERS):
            fused = None
            ms, fused = timeit(lambda: model.get_vl_embeddings(
                images=images, image_mask=image_mask, prompt=prompt))
            vl.append(ms)
            ms, st_t = timeit(lambda: model.prepare_state(norm_state))
            st.append(ms)
            ms, _ = timeit(lambda: model.predict_action(fused, st_t, action_mask=action_mask))
            ac.append(ms)
            ms, _ = timeit(lambda: model.run_inference(
                images=images, image_mask=image_mask, prompt=prompt,
                state_input=norm_state, action_mask=action_mask))
            tot.append(ms)

    out = {"label": label, "mask_len": mask_len, "iters": ITERS, "warmup": WARMUP,
           "vl_embeddings_ms": stats(vl), "prepare_state_ms": stats(st),
           "predict_action_ms": stats(ac), "run_inference_ms": stats(tot),
           "sum_of_stages_mean": round(statistics.mean(vl) + statistics.mean(st) + statistics.mean(ac), 2)}
    print(json.dumps(out, indent=2))
    del model
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    results = [run(1, "benchmark-compatible image_mask=zeros(1)"),
               run(3, "LIBERO client image_mask=[1,1,0]")]
    p = os.path.join(os.path.dirname(__file__), "..", "results",
                     "jp7-evo-depth-breakdown-20261110.json")
    json.dump(results, open(p, "w"), indent=2)
    print("written", os.path.abspath(p))
