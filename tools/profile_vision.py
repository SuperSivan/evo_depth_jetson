"""Split get_vl_embeddings into its four stages so we know what to optimise."""
import json, os, sys, time, statistics
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evo_depth")))
from scripts.Evo_depth_server import load_model_and_normalizer

CKPT = os.environ.get("EVO_DEPTH_CKPT_DIR", "/home/nvidia/evo_weights/lib/libero_spatial")
ITERS = int(os.environ.get("ITERS", "5"))
WARMUP = int(os.environ.get("WARMUP", "2"))


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(statistics.mean(xs), 2), "p50": round(statistics.median(xs), 2)}


def main():
    model, normalizer = load_model_and_normalizer(CKPT)
    emb = model.embedder
    g = torch.Generator(device="cpu").manual_seed(1234)
    images = [torch.rand((3, 448, 448), generator=g).to("cuda") for _ in range(3)]
    image_mask = torch.zeros((3,), dtype=torch.int32, device="cuda")
    prompt = "dummy prompt"

    stages = {k: [] for k in ["preprocess", "vit", "da3", "fuse_prompt", "language", "film"]}
    shapes = {}

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for it in range(WARMUP + ITERS):
            ms, (pv, ntl) = timed(lambda: emb._preprocess_images(images))
            shapes["pixel_values"] = list(pv.shape)
            shapes["num_tiles"] = ntl
            ms2, vit = timed(lambda: emb.model.extract_feature(pv))
            shapes["vit_embeds"] = list(vit.shape)
            ms3, dpt = timed(lambda: checkpoint(lambda x: emb.dpt_embedder(x, image_mask),
                                                images, use_reentrant=False))
            # _build_multimodal_prompt sets img_context_token_id and expands <image>
            prompt_full = emb._build_multimodal_prompt(ntl, prompt)
            ms4, (inputs_embeds, attn_mask) = timed(
                lambda: emb._prepare_and_fuse_embeddings(prompt_full, vit, image_mask, ntl))
            shapes["inputs_embeds"] = list(inputs_embeds.shape)
            ms5, out = timed(lambda: emb.model.language_model(
                inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                output_hidden_states=True, return_dict=True))
            hidden = out.hidden_states[-1]
            ms6, _ = timed(lambda: emb.dpt_linear_film(dpt))
            if it >= WARMUP:
                for k, v in zip(stages, [ms, ms2, ms3, ms4, ms5, ms6]):
                    stages[k].append(v)

    res = {k: stats(v) for k, v in stages.items()}
    res["_shapes"] = shapes
    res["_total_mean"] = round(sum(v["mean"] for v in res.values() if isinstance(v, dict) and "mean" in v), 2)
    print(json.dumps(res, indent=2))
    p = os.path.join(os.path.dirname(__file__), "..", "results", "jp7-evo-depth-vision-profile.json")
    json.dump(res, open(p, "w"), indent=2)
    print("written", os.path.abspath(p))


main()
