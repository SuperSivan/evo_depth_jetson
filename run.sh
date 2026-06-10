#!/bin/bash
# Evo-Depth LIBERO 评估 — 在两个终端分别执行以下命令
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# 终端 1：模型 WebSocket 服务（端口 9000）
# cd Evo_depth && python scripts/Evo_depth_server.py
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 终端 2：LIBERO 仿真客户端（可选 Web 推流）                                │
# └─────────────────────────────────────────────────────────────────────────┘
# cd simulator_evaluation/LIBERO-evaluation \
#   && bash ./test_libero.sh ./logs libero_spatial ws://127.0.0.1:9000 8080
#
# test_libero.sh 参数:
#   $1  log_path     日志与视频目录，例如 ./logs
#   $2  task         libero_spatial | libero_goal | libero_object | libero_10
#   $3  server_url   WebSocket 地址，默认 ws://127.0.0.1:9000
#   $4  stream_port  Web 推流端口，例如 8080；留空则关闭推流
#
# Web 推流（传第 4 个参数 8080 时启用）:
#   直播页面  http://127.0.0.1:8080/
#   视频流    http://127.0.0.1:8080/video
#   状态 JSON http://127.0.0.1:8080/status
#
# 离线视频保存: simulator_evaluation/LIBERO-evaluation/logs/videos/<task>/
#
# 不启用 Web 推流时，去掉第 4 个参数:
#   bash ./test_libero.sh ./logs libero_spatial ws://127.0.0.1:9000
