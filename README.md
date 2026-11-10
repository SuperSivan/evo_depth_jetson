# EvoDepth — Jetson Orin Nano 部署

Evo-Depth 是轻量级的深度增强视觉-语言-动作（VLA）策略，由视觉语言主干、隐式深度编码模块（IDEM）与空间增强模块（SEM）组成，用 flow-matching 动作专家输出连续动作块，输入是多视角 RGB、机器人状态和语言指令。

原项目 [MINT-SJTU/Evo-Depth](https://github.com/MINT-SJTU/Evo-Depth) · 论文 [arXiv 2605.14950](https://arxiv.org/abs/2605.14950)

本仓库是它的 Jetson 适配版。上游完整的说明（LIBERO / LIBERO-PLUS / MetaWorld / VLA-Arena 评测、Progressive Alignment 训练配方）保留在 [README.upstream-jp6.md](README.upstream-jp6.md)，本文件只写在本机上跑推理需要知道的部分。

## 环境

| 项 | 实测值 |
|---|---|
| 板卡 | Jetson Orin Nano Engineering Reference Developer Kit Super，7.5 GB 统一内存 |
| 系统 | JetPack 7（L4T R39.2.1），Ubuntu 24.04.4，Python 3.12.3 |
| CUDA | 13.2 |
| PyTorch | 2.15.0+cu132，PyPI 官方 aarch64 轮子 |

上游假设的是 JetPack 6 / Python 3.10 / torch 2.5。这台板子上没有可用的 `sm_87` Jetson 轮子：
`pypi.jetson-ai-lab.io/sbsa/cu130` 的 `get_arch_list()` 只有 `sm_110` 和 `sm_121`，
`jp6/*` 索引只有 cp310。PyPI 的 aarch64 轮子带 `sm_80`，靠 major 版本的二进制兼容能在 `sm_87` 上执行。

## 安装

```bash
cd ~/evo_depth_jetson
python3 -m venv .venv-py
source .venv-py/bin/activate
pip install -U pip
pip install torch torchvision          # 自带整套 CUDA 13 aarch64 运行库
pip install -r requirements-jp7.txt    # 本机实测可用的锁定版本
```

不编 `flash-attn`、`gsplat`、`pycolmap`。前两个在 8 GB 内存的板子上编译不可行，
后者的 import 链已经改成惰性加载，改动见 [patches/jp7-deploy.patch](patches/jp7-deploy.patch)。

## 权重

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 本机直连 huggingface.co 不稳定
export HF_HUB_DISABLE_XET=1                # Xet 后端会报 Errno 101 Network is unreachable

hf download OpenGVLab/InternVL3-1B     --local-dir ~/evo_weights/InternVL3-1B
hf download depth-anything/da3-base    --local-dir ~/evo_weights/da3-base
hf download MINT-SJTU/EVO-Depth-LIBERO --include "libero_spatial/**" --local-dir ~/evo_weights/lib

mkdir -p weights
ln -sfn ~/evo_weights/InternVL3-1B weights/InternVL3-1B
ln -sfn ~/evo_weights/da3-base    weights/da3-base
```

`weights/` 下必须是软链。`Evo_depth_server.py:62` 会把 config 里非绝对路径的 `vlm_name`
和 `da3_model_path` 解析成相对仓库根的路径，检查点 `config.json` 里这两个键已改写为
`weights/InternVL3-1B` 与 `weights/da3-base`。

## 运行

```bash
source .venv-py/bin/activate
EVO_DEPTH_CKPT_DIR=~/evo_weights/lib/libero_spatial \
  python -u Evo_depth/scripts/Evo_depth_server.py     # ws://0.0.0.0:9000
```

另开一个终端验证：

```bash
python tools/smoke_client.py ws://127.0.0.1:9000      # 期望 (50, 24) 动作块
```

协议注意点：`action_mask` 是 **24 个逐维开关**（`[1]*7 + [0]*17`），不是标量，传标量会在
`flow_matching.py:343` 崩。`image_mask` 是 3 个视角开关，LIBERO 客户端用 `[1,1,0]`。

## 测速

测速前先停掉服务，7.5 GB 内存装不下两份模型。`benchmark_max_perf.sh` 在本机跑不通，
它第一步 `sudo nvpmodel -m` 就需要密码；确认 `nvpmodel -q` 报 `MAXN_SUPER`、
CPU 频率 `scaling_cur_freq` 等于 `scaling_max_freq` 后直接调脚本即可。

```bash
python -u Evo_depth/scripts/benchmark_evo_depth.py \
  --ckpt_dir ~/evo_weights/lib/libero_spatial \
  --num_warmup 5 --num_iterations 30 --input_mode random

python tools/bench_breakdown.py     # 分段耗时
```

本机实测（`libero_spatial`，50 步采样，bf16，MAXN_SUPER 且时钟已锁）：

| 指标 | 数值 |
|---|---|
| 平均 / 中位 / P99 延迟 | 2033.25 / 2033.04 / 2042.52 ms |
| 吞吐 | 0.49 FPS |
| 显存 平均 / 峰值 | 2631 / 3152 MB |
| 功耗 平均 / 峰值 | 7.59 / 7.63 W |
| 分段占比 | vision(InternVL3-1B+DA3) 1276 ms / 60.6%，动作头 831 ms / 39.5% |

websocket 往返 3.749 s，其中 1.716 s 是 9.5 MB JSON 的编解码与传输。
分段明细与对照见 [DEPLOY_NOTES.md](DEPLOY_NOTES.md) 第 5、7 节，原始数据在 `results/`。

## 文档与产物

| 文件 | 内容 |
|---|---|
| [DEPLOY_NOTES.md](DEPLOY_NOTES.md) | 环境选型、权重、协议细节、实测数据与限制 |
| [ACCESS.md](ACCESS.md) | 板子的访问链路 |
| [patches/jp7-deploy.patch](patches/jp7-deploy.patch) | 全部代码改动（可回滚，各文件旁有 `.orig`） |
| `results/` | 测速结果 JSON |
| `tools/` | 冒烟客户端与分段测速脚本 |
