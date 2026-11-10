"""InternViT feature extraction through a TensorRT engine.

The engine is built with a static 1x3x448x448 input, so callers pass one tile at a
time. Buffers are addressed through torch tensor pointers; no cuda bindings needed.
"""
import sys

sys.path.append("/usr/lib/python3.12/dist-packages")
import tensorrt as trt
import torch

_IN = "pixel_values"
_OUT = "vit_embeds"
_cache = {}


class TRTExtractFeature:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize engine {engine_path}")
        self.ctx = self.engine.create_execution_context()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            assert (name, int(mode)) in ((_IN, int(trt.TensorIOMode.INPUT)), (_OUT, int(trt.TensorIOMode.OUTPUT))), \
                f"unexpected binding {name} mode={mode}"
            if name == _OUT:
                self.out_shape = tuple(shape)

    def __call__(self, pixel_values, dtype=torch.bfloat16):
        x = pixel_values.unsqueeze(0) if pixel_values.dim() == 3 else pixel_values
        x = x.contiguous().float()
        if tuple(x.shape) != (1, 3, 448, 448):
            raise ValueError(f"engine expects 1x3x448x448, got {tuple(x.shape)}")
        out = torch.empty(self.out_shape, dtype=torch.float32, device=x.device)
        self.ctx.set_tensor_address(_IN, x.data_ptr())
        self.ctx.set_tensor_address(_OUT, out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TRT execute failed")
        torch.cuda.current_stream().synchronize()
        return out.to(dtype)


def get(engine_path):
    if engine_path not in _cache:
        _cache[engine_path] = TRTExtractFeature(engine_path)
    return _cache[engine_path]


def extract_features(pixel_values, engine_path):
    """pixel_values: [T,3,448,448] -> [T,256,896] using one engine call per tile."""
    fn = get(engine_path)
    dtype = pixel_values.dtype
    outs = [fn(pixel_values[i], dtype=dtype) for i in range(pixel_values.shape[0])]
    return torch.cat(outs, dim=0)
