#!/bin/bash
# Feed the same input through the TRT engine and compare with the PyTorch reference.
set -e
cd ~/evo_depth_jetson/export
source ~/evo_depth_jetson/.venv-py/bin/activate

# npy -> raw f32 for trtexec
python - <<'PY'
import numpy as np
x = np.load("extract_feature_in.npy").astype(np.float32)
x.tofile("extract_feature_in.bin")
print("input bin:", x.shape, x.size * 4, "bytes")
PY

rm -rf trt_out && mkdir -p trt_out
/usr/bin/trtexec \
  --loadEngine=internvit_fp16.engine \
  --loadInputs=pixel_values:extract_feature_in.bin \
  --exportOutput=trt_out \
  --warmUp=500 --iterations=20 --avgRuns=10 --duration=0 \
  > /home/nvidia/evo_trtexec_parity.log 2>&1 || true

grep -E "GPU Compute Time|Throughput" /home/nvidia/evo_trtexec_parity.log | tail -4
python - <<'PY'
import numpy as np, glob, os
ref = np.load("extract_feature_ref.npy").astype(np.float32).ravel()
cands = glob.glob("trt_out/*.bin") + glob.glob("trt_out/*/*.bin")
print("exported:", [os.path.basename(c) for c in cands])
best = None
for c in cands:
    a = np.fromfile(c, dtype=np.float32)
    if a.size == ref.size:
        best = (c, a)
if best is None:
    print("PARITY_FAIL: no exported tensor matches reference shape", ref.size)
    raise SystemExit(1)
c, a = best
d = np.abs(a - ref)
cos = float(np.dot(a, ref) / (np.linalg.norm(a) * np.linalg.norm(ref) + 1e-12))
print(f"tensor: {os.path.basename(c)}")
print(f"  max_abs={d.max():.6g}  mean_abs={d.mean():.6g}  cosine={cos:.8f}")
print(f"  ref  range [{ref.min():.4f}, {ref.max():.4f}]  trt range [{a.min():.4f}, {a.max():.4f}]")
print("PARITY_OK" if cos > 0.999 else "PARITY_WARN")
PY
