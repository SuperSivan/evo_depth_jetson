# evo_depth_jetson 在 Orin Nano (JP7) 上的部署记录

部署日期 2026-11-09。目标板：`nvidia@172.25.80.139`，经 124 跳板访问。
板端仓库路径 `/home/nvidia/evo_depth_jetson`。

## 1. 环境与仓库预期不符

| 项 | 仓库预期 | 板子实际 |
|---|---|---|
| JetPack | 6.x | **7 (R39.2.1)** |
| Ubuntu / Python | 22.04 / 3.10 | **24.04 / 3.12.3** |
| CUDA | 12.x | **13.2** |
| PyTorch | 2.5.0 (jetson 轮子) | **2.15.0+cu132 (PyPI aarch64)** |

### 选轮子的过程（两个死胡同）

- `pypi.jetson-ai-lab.io/sbsa/cu130` 的 torch 2.9.1：`get_arch_list()` 只有 `sm_110, sm_121`，
  在 Orin (`sm_87`) 上 bf16 matmul 直接 CUDA kernel error。它只给 Thor/Blackwell。
  该轮子还依赖系统 NVPL (lapack/blas/common) 与 cuDSS，其中 cuDSS 需 0.7.x（提供 `libcudss.so.0`，
  1.0.0 只有 `.so.1`）。补齐后仍跑不了，链接层的努力全部作废。
- `jp6/cu128` 的 torch 2.7.1：只有 cp310 轮子，py3.12 装不上。

### 走通的路

PyPI 官方 aarch64 轮子 `torch-2.15.0-cp312-cp312-manylinux_2_28_aarch64.whl`（483.7 MB），
自带整套 CUDA 13 aarch64 运行库（cublas 13.4 / cudnn cu13 9.26 / nccl / cusparse / cusolver /
cuda_runtime 13.2），不需要系统 CUDA 或 NVPL：

```
torch 2.15.0+cu132  cuda 13.2
arch ['sm_80','sm_90','sm_100','sm_110','sm_120']
avail True  dev Orin  bf16 matmul ok True
```

`sm_87` 不在列表里但能跑，因为 CUDA 二进制兼容按大版本走，`sm_80` 的 cubin 可在 major 8 的
`sm_86/sm_87` 上执行。这就是它和 `sm_110` 轮子的本质区别。

## 2. 代码改动（两处，均可回滚）

1. `Evo_depth/model/internvl3/internvl3_embedder.py:83` 和
   `internvl3_embedder_da3.py:94`：`use_flash_attn=True` → `False`。
   板端不编 flash-attn（8G 内存编译不可行），InternVL3 回落 torch SDPA。运行时日志
   会打印 `FlashAttention2 is not installed.`，这是预期行为。
2. `Evo_depth/model/depth_anything_3/utils/export/__init__.py`：把 `from .gs import ...`
   和 `from .colmap import ...` 等模块级 import 移入 `export()` 的各分支，改惰性加载。
   原因：`api.py:34` 在模块顶层就 import 了整棵 export 子树，导致推理被迫依赖
   `moviepy`（3DGS 视频编码）和 `pycolmap`（需编 COLMAP + Ceres，aarch64 py3.12 上基本不可行）。
   原文件备份为 `__init__.py.orig`。不用的导出格式行为不变。

## 3. 权重

| 内容 | 大小 | 位置 |
|---|---|---|
| InternVL3-1B | 1.8G | `~/evo_weights/InternVL3-1B` |
| EVO-Depth-LIBERO / libero_spatial | 1.8G | `~/evo_weights/lib/libero_spatial` |
| depth-anything/da3-base | 517M | `~/evo_weights/da3-base` |

仓库根下 `weights/InternVL3-1B`、`weights/da3-base` 是指向它们的符号链接，因为
`Evo_depth_server.py:62` 会把 config 里非绝对路径的 `vlm_name` / `da3_model_path`
解析成相对 `repo_root` 的路径。checkpoint 的 config.json 里这两个键已改写为 `weights/...`。

板子直连 huggingface.co 不稳定（`api=000`），下载统一走 `HF_ENDPOINT=https://hf-mirror.com`
加 `HF_HUB_DISABLE_XET=1`（Xet 后端会报 `Errno 101 Network is unreachable`）。

## 4. 协议注意点

客户端 payload 的 `action_mask` 不是标量，是 **24 个逐维开关**（`libero_client.py:160`
用 `[1]*7 + [0]*17`）。传标量会让服务端在 `flow_matching.py:343` 崩：
`RuntimeError: shape '[1, 1, 24]' is invalid for input of size 1`。
`image_mask` 是 3 个视角开关，仓库的 LIBERO 客户端用 `[1,1,0]`（第三人称 + 腕部）。

## 5. 实测结果

启动：`EVO_DEPTH_CKPT_DIR=~/evo_weights/lib/libero_spatial python -u Evo_depth/scripts/Evo_depth_server.py`，
模型加载后常驻内存 3.6 GB / 7.5 GB。

冒烟（随机观测，3 图 448x448，双相机 mask）：

```
round-trip 3.749 s
actions shape (50, 24) float32, finite=True, min=-0.3594 max=1.0157
```

测速（`benchmark_evo_depth.py --num_warmup 5 --num_iterations 30 --input_mode random`，
MAXN_SUPER 且 CPU 1728MHz / GPU 1020MHz 已锁频）：

| 指标 | 本次 (JP7 / torch 2.15) | 仓库 docs 基线 (JP6 / torch 2.5) |
|---|---:|---:|
| 平均延迟 | **2033.25 ms** | 5524.94 ms |
| 中位数 | 2033.04 ms | 5528.21 ms |
| P99 | 2042.52 ms | 5568.03 ms |
| 标准差 | 3.79 ms | 13.89 ms |
| 吞吐 | **0.49 FPS** | 0.18 FPS |
| 平均显存 | 2631.47 MB | 2751.82 MB |
| 峰值显存 | 3151.97 MB | 3272.30 MB |
| 平均功耗 | 7.59 W | 6.22 W |

延迟约为基线的 1/2.7，显存持平。

冒烟往返 3.749 s 比基准的 2.033 s 多出的部分基本是协议开销：3 张 448x448x3 的
嵌套 list 拼成 9.5 MB JSON，在 Orin 上编解码约 1.7 s。真实部署应换二进制传输。

## 6. 未完成

- LIBERO 仿真评测（`simulator_evaluation/LIBERO-evaluation/test_libero.sh`）未跑。
  需要 MuJoCo + robosuite + LIBERO 全套，且客户端只在 websocket 一端送观测，
  可以放在 x86 侧跑、连板子的 9000 端口，避免在 8G 板子上装仿真环境。

## 7. 分段耗时与结论更新（2026-11-10）

`benchmark_evo_depth.py` 只给总时延，看不出钱花在哪。`tools/bench_breakdown.py`
把 `run_inference` 的三段分别计时，两种 `image_mask` 口径各跑 10 轮
（结果在 `results/jp7-evo-depth-breakdown-20261110.json`）：

| 阶段 | p50 | 占总时比例 |
|---|---:|---:|
| `get_vl_embeddings`（InternVL3-1B + DA3） | **1276.35 ms** | **60.6%** |
| `predict_action`（50 步 flow matching） | **831.36 ms** | **39.5%** |
| `prepare_state` | 0.05 ms | 0.0% |
| 三段之和 | 2107.76 ms | |
| `run_inference` 总时（同一脚本内） | 2106.88 ms | |

三段之和与总时差 −0.9 ms，说明没有隐藏的拼接开销。这份脚本里总时 2106.88 ms
比正式基准的 2033.25 ms 高 3.6%，原因是逐段计时每段都多一次 `synchronize` 和分配抖动，
**正式数字仍以 2033.25 ms 为准**。

两条由数据得出的结论：

1. **vision 是主项，占 60.6%**。50 步 flow matching 平均 16.63 ms/步，合计 831 ms。
   要继续压时间，先动 InternVL3-1B 与 DA3 那一路，而不是采样步数。
2. **mask 口径不是问题**。仓库基准用 `zeros(1)`，LIBERO 客户端用 `[1,1,0]`，
   实测两者 vl 1276.35 对 1277.00、action 831.36 对 828.99，属噪声。
   之前担心的「基准与真实推理口径不一致」不成立。

仍未测：精度（随机输入，无参考动作），以及 LIBERO 仿真评测。
