"""Split the DA3 stage into host round trip and the DPT network itself."""
import json, os, sys, time, statistics
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evo_depth")))
from scripts.Evo_depth_server import load_model_and_normalizer

CKPT = os.environ.get("EVO_DEPTH_CKPT_DIR", "/home/nvidia/evo_weights/lib/libero_spatial")
ITERS = 3


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def main():
    model, _ = load_model_and_normalizer(CKPT)
    da3 = model.embedder.dpt_embedder
    api = da3.model
    g = torch.Generator(device="cpu").manual_seed(3)
    images = [torch.rand((3, 448, 448), generator=g).to("cuda") for _ in range(3)]
    mask = torch.zeros((3,), dtype=torch.int32, device="cuda")

    full, host, net = [], [], []
    with torch.no_grad():
        for _ in range(ITERS):
            ms, _ = timed(lambda: da3(images, mask))
            full.append(ms)

            def to_pil():
                x_list = []
                for i in range(len(images)):
                    t = images[i].cpu()
                    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    x_list.append(Image.fromarray(arr).convert("RGB"))
                return x_list
            ms, x_list = timed(to_pil)
            ms2, (imgs_cpu, ext, intr) = timed(
                lambda: api._preprocess_inputs(x_list, None, None, 504, "upper_bound_resize"))
            ms3, (imgs, _, _) = timed(lambda: api._prepare_model_inputs(imgs_cpu, ext, intr))
            ms4, _ = timed(lambda: api.model._get_dpt_embedding(imgs))
            host.append(ms + ms2 + ms3)
            net.append(ms4)

    out = {"full_p50": round(statistics.median(full), 2),
           "host_roundtrip_p50": round(statistics.median(host), 2),
           "dpt_network_p50": round(statistics.median(net), 2)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "jp7-evo-depth-da3-profile.json"), "w"), indent=2)


main()
