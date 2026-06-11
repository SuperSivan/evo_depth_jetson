#!/usr/bin/env python3
"""
Evo_depth 模型测速脚本
用于在 Jetson Orin Nano 等设备上测量推理延迟和吞吐量
"""

import sys
import os
import time
import re
import shutil
import subprocess
import numpy as np
import torch
from torchvision import transforms

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - CPU metrics will not be shown")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo_depth import Evo_depth


def load_model_and_normalizer(ckpt_dir):
    import json
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    # Resolve relative paths in config
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for key in ["vlm_name", "da3_model_path"]:
        if key in config and isinstance(config[key], str) and not config[key].startswith("/"):
            config[key] = os.path.join(repo_root, config[key])

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = 50

    model = Evo_depth(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    model = model.to("cuda")

    # 使用真实的 Normalizer 类（从 Evo_depth_server.py 复制）
    class Normalizer:
        def __init__(self, stats_or_path):
            if isinstance(stats_or_path, str):
                with open(stats_or_path, "r") as f:
                    stats = json.load(f)
            else:
                stats = stats_or_path

            def pad_to_24(x):
                x = torch.tensor(x, dtype=torch.float32)
                if x.shape[0] < 24:
                    pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                    x = torch.cat([x, pad], dim=0)
                elif x.shape[0] > 24:
                    raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
                return x

            if len(stats) != 1:
                raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

            robot_key = list(stats.keys())[0]
            robot_stats = stats[robot_key]

            self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
            self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
            self.action_min = pad_to_24(robot_stats["action"]["min"])
            self.action_max = pad_to_24(robot_stats["action"]["max"])

        def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
            state_min = self.state_min.to(state.device, dtype=state.dtype)
            state_max = self.state_max.to(state.device, dtype=state.dtype)
            return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

        def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
            action_min = self.action_min.to(action.device, dtype=action.dtype)
            action_max = self.action_max.to(action.device, dtype=action.dtype)
            if action.ndim == 1:
                action = action.view(1, -1)
            return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min
    normalizer = Normalizer(stats)

    return model, normalizer


def create_dummy_input(device="cuda"):
    """创建输入数据用于测速，支持随机图像或真实图片"""
    return create_benchmark_input(device=device, input_mode="random", image_paths=None)


class GPUMonitor:
    def __init__(self):
        self.backend = None
        self.nvml_handle = None
        self.tegrastats_supports_count = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.backend = "nvml"
            except Exception:
                self.nvml_handle = None

        if self.backend is None and shutil.which("tegrastats") is not None:
            self.backend = "tegrastats"
            try:
                help_out = subprocess.run(
                    ["tegrastats", "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                help_text = (help_out.stdout or "") + "\n" + (help_out.stderr or "")
                self.tegrastats_supports_count = "--count" in help_text
            except Exception:
                self.tegrastats_supports_count = False

    def sample_power_w(self):
        if self.backend == "nvml" and self.nvml_handle is not None:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
            except Exception:
                return None

        if self.backend == "tegrastats":
            try:
                if self.tegrastats_supports_count:
                    cmd = ["tegrastats", "--interval", "1000", "--count", "1"]
                elif shutil.which("timeout") is not None:
                    cmd = ["timeout", "2s", "tegrastats", "--interval", "1000"]
                else:
                    return None

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3,
                )
                line = (result.stderr or "") + " " + (result.stdout or "")
                match = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
                if match:
                    return float(match.group(1)) / 1000.0
            except Exception:
                return None

        return None

    def close(self):
        if self.backend == "nvml" and self.nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def create_benchmark_input(device="cuda", input_mode="random", image_paths=None, image_size=448):
    """创建输入数据用于测速。

    input_mode:
        - random: 使用随机张量
        - real: 从 image_paths 读取真实图片
    """
    if input_mode not in {"random", "real"}:
        raise ValueError(f"Unsupported input_mode: {input_mode}. Use 'random' or 'real'.")

    if input_mode == "random":
        images = [
            torch.rand((3, image_size, image_size), dtype=torch.float32, device=device)
            for _ in range(3)
        ]
    else:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for real image input. Please install pillow.")
        if not image_paths:
            raise ValueError("input_mode='real' requires --image_paths")

        preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        loaded_images = []
        for path in image_paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = preprocess(img).to(device=device, dtype=torch.float32)
                loaded_images.append(tensor)

        if len(loaded_images) == 0:
            raise ValueError("No valid images loaded from --image_paths")

        # 统一为 3 视角输入：不足补最后一张，超出截断
        if len(loaded_images) < 3:
            loaded_images.extend([loaded_images[-1]] * (3 - len(loaded_images)))
        images = loaded_images[:3]

    # 模拟状态向量 (24维，与 Evo_depth_server.py 一致)
    state = torch.zeros((1, 7), dtype=torch.float32, device=device)
    # 填充到 24 维
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    
    # 模拟其他输入
    prompt = "dummy prompt"
    image_mask = torch.zeros((1,), dtype=torch.int32, device=device)
    # action_mask 需要是 (1, 24)，对应 per_action_dim=24
    # 前7个1对应真实动作维度，后面17个0是padding
    action_mask_data = [1] * 7 + [0] * 17
    action_mask = torch.tensor([action_mask_data], dtype=torch.int32, device=device)
    
    return {
        "images": images,
        "state": state,
        "prompt": prompt,
        "image_mask": image_mask,
        "action_mask": action_mask
    }


def benchmark_model(model, normalizer, inputs, num_warmup=5, num_iterations=100, monitor_interval=5):
    """
    运行模型测速
    
    Args:
        model: Evo_depth 模型
        normalizer: 状态和动作归一化器
        inputs: 输入数据字典
        num_warmup: 预热轮次
        num_iterations: 测速轮次
    """
    device = "cuda"
    
    print(f"开始测速...")
    print(f"预热轮次: {num_warmup}")
    print(f"测速轮次: {num_iterations}")
    
    # 对 state 进行 normalize
    norm_state = normalizer.normalize_state(inputs["state"]).to(dtype=torch.float32)
    
    # 预热
    print("\n[1/3] 预热中...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(num_warmup):
            model.run_inference(
                images=inputs["images"],
                image_mask=inputs["image_mask"],
                prompt=inputs["prompt"],
                state_input=norm_state,
                action_mask=inputs["action_mask"]
            )
    
    # 测速
    print("\n[2/3] 测速中...")
    torch.cuda.synchronize()
    times = []
    power_samples_w = []
    memory_samples_mb = []
    gpu_monitor = GPUMonitor()

    if gpu_monitor.backend is None:
        print("⚠️ 未检测到 NVML/tegrastats，功耗监控将被跳过")
    else:
        print(f"GPU 监控后端: {gpu_monitor.backend}")

    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(num_iterations):
            start_time = time.time()
            torch.cuda.synchronize()
            
            model.run_inference(
                images=inputs["images"],
                image_mask=inputs["image_mask"],
                prompt=inputs["prompt"],
                state_input=norm_state,
                action_mask=inputs["action_mask"]
            )
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

            current_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_samples_mb.append(current_mem_mb)

            if monitor_interval > 0 and ((i + 1) % monitor_interval == 0):
                power_w = gpu_monitor.sample_power_w()
                if power_w is not None:
                    power_samples_w.append(power_w)
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{num_iterations} 轮")
    
    # 统计结果
    print("\n[3/3] 统计结果...")
    times = np.array(times) * 1000  # 转换为毫秒
    
    avg_time = np.mean(times)
    median_time = np.median(times)
    p99_time = np.percentile(times, 99)
    std_time = np.std(times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_mem_mb = float(np.mean(memory_samples_mb)) if memory_samples_mb else 0.0
    avg_power_w = float(np.mean(power_samples_w)) if power_samples_w else None
    max_power_w = float(np.max(power_samples_w)) if power_samples_w else None

    gpu_monitor.close()
    
    print("\n" + "="*60)
    print("Evo_depth 模型测速结果")
    print("="*60)
    print(f"平均延迟:    {avg_time:.2f} ms")
    print(f"中位数延迟:   {median_time:.2f} ms")
    print(f"P99 延迟:    {p99_time:.2f} ms")
    print(f"标准差:      {std_time:.2f} ms")
    print(f"吞吐量:      {fps:.2f} FPS")
    print(f"平均显存:     {avg_mem_mb:.2f} MB")
    print(f"峰值显存:     {peak_mem_mb:.2f} MB")
    if avg_power_w is not None:
        print(f"平均功耗:     {avg_power_w:.2f} W")
        print(f"最大功耗:     {max_power_w:.2f} W")
    else:
        print("平均功耗:     N/A (未采样)")
    print("="*60)
    
    return {
        "avg_ms": avg_time,
        "median_ms": median_time,
        "p99_ms": p99_time,
        "std_ms": std_time,
        "fps": fps,
        "avg_mem_mb": avg_mem_mb,
        "peak_mem_mb": peak_mem_mb,
        "avg_power_w": avg_power_w,
        "max_power_w": max_power_w,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evo_depth 模型测速脚本")
    parser.add_argument(
        "--ckpt_dir", 
        type=str, 
        default=None,
        help="模型权重目录 (包含 config.json, norm_stats.json, mp_rank_00_model_states.pt)"
    )
    parser.add_argument(
        "--num_warmup", 
        type=int, 
        default=5,
        help="预热轮次 (默认为 5)"
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=10,
        help="测速轮次 (默认为 100)"
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="random",
        choices=["random", "real"],
        help="输入图像模式: random(随机张量) 或 real(真实图片)"
    )
    parser.add_argument(
        "--image_paths",
        nargs="+",
        default=None,
        help="真实图片路径列表（input_mode=real 时生效）"
    )
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=5,
        help="功耗采样间隔（每 N 轮采样一次，0 表示关闭功耗采样）"
    )
    
    args = parser.parse_args()
    
    # 确定 ckpt_dir
    if args.ckpt_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        args.ckpt_dir = os.path.join(
            repo_root,
            "simulator_evaluation",
            "LIBERO-evaluation",
            "libero_cfgs",
            "libero_spatial",
        )
    
    print(f"使用权重目录: {args.ckpt_dir}")
    
    # 加载模型
    print("\n加载模型中...")
    model, normalizer = load_model_and_normalizer(args.ckpt_dir)
    print("模型加载完成!")
    
    # 创建输入
    print(f"输入模式: {args.input_mode}")
    if args.input_mode == "real":
        print(f"真实图片路径: {args.image_paths}")
    inputs = create_benchmark_input(
        device="cuda",
        input_mode=args.input_mode,
        image_paths=args.image_paths,
    )
    
    # 运行测速
    benchmark_model(
        model,
        normalizer,
        inputs,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        monitor_interval=args.monitor_interval,
    )


if __name__ == "__main__":
    main()
