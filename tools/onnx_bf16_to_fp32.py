"""Rewrite bf16 initializers as fp32 (bit-exact: bf16 is truncated fp32)."""
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)

def bf16_bytes_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)

n = 0
for t in m.graph.initializer:
    if t.data_type == TensorProto.BFLOAT16:
        arr = bf16_bytes_to_f32(t.raw_data).reshape(list(t.dims))
        t.CopyFrom(numpy_helper.from_array(arr, t.name))
        n += 1

casts = [nd for nd in m.graph.node if nd.op_type == "Cast"
         and nd.attribute[0].i == TensorProto.BFLOAT16]
print(f"converted {n} initializers; Cast->bf16 nodes: {len(casts)}")
onnx.save(m, dst)
print("saved", dst)
