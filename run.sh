#!/bin/bash
# Evo-Depth 在本机（Orin Nano / JetPack 7）的启动顺序，已在 2026-11-09 实测通过。
#
# 阶段一：装环境（只做一次）
#   python3 -m venv .venv-py && source .venv-py/bin/activate
#   pip install -r requirements-jp7.txt
#   权重与软链见 README.md 的「Jetson 部署」一节。
#
# 阶段二：起服务（终端 1）
#   cd ~/evo_depth_jetson
#   source .venv-py/bin/activate
#   EVO_DEPTH_CKPT_DIR=~/evo_weights/lib/libero_spatial \
#     python -u Evo_depth/scripts/Evo_depth_server.py
#   就绪标志：Evo_depth server running at ws://0.0.0.0:9000（常驻约 3.6 GB）
#
# 阶段三：验证推理（终端 2）
#   source .venv-py/bin/activate
#   python tools/smoke_client.py ws://127.0.0.1:9000
#   期望：actions shape (50, 24)，往返约 3.7 s，其中 1.7 s 是 9.5 MB JSON 的编解码
#
# 阶段四：测速（先停掉服务，7.5 GB 内存装不下两份模型）
#   python -u Evo_depth/scripts/benchmark_evo_depth.py \
#     --ckpt_dir ~/evo_weights/lib/libero_spatial \
#     --num_warmup 5 --num_iterations 30 --input_mode random
#   python tools/bench_breakdown.py     # 分段耗时
#
# 阶段五（未跑）：LIBERO 仿真评测。客户端只在 websocket 一端送观测，
#   可以放在 x86 机器上跑，连本机的 9000 端口，不必在 8 GB 板子上装 MuJoCo 全套。
#   cd simulator_evaluation/LIBERO-evaluation
#   bash ./test_libero.sh ./logs libero_spatial ws://<board-ip>:9000
#
# 注意：benchmark_max_perf.sh 在本机跑不通，它第一步 sudo nvpmodel -m 需要密码。
# 时钟与电源模式另行确认，条件见 README.md「Jetson 推理测速」。

echo "run.sh 是说明文件，不自动执行任何步骤。见上方的五个阶段。"
