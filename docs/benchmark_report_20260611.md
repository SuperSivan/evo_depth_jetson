# Evo_depth 算法测速报告（2026-06-11）

## 1. 测试目的

对当前仓库中的 `Evo_depth` 推理链路进行端到端测速，输出时延、吞吐、显存与功耗指标，为后续优化提供基线。

## 2. 测试环境

- 设备：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- 运行目录：仓库根目录 `./`
- 功耗模式：`MAXN_SUPER`
- 频率状态：`jetson_clocks` 已锁频
  - CPU：6 核 `1728 MHz`（Min=Max）
  - GPU：`1020 MHz`（Min=Max）
  - EMC：当前 `2133 MHz`，`FreqOverride=1`

> 以上来自测试后采集：`sudo nvpmodel -q --verbose` 与 `sudo jetson_clocks --show`。

## 3. 测试命令与参数

实际执行命令：

```bash
bash benchmark_max_perf.sh -- --num_warmup 5 --num_iterations 30 --input_mode random
```

关键参数：

- 预热轮次：5
- 正式测速轮次：30
- 输入模式：`random`
- 监控后端：`tegrastats`（脚本输出显示）

原始日志已保存：`docs/benchmark_run_20260611.txt`

## 4. 实测结果

来自脚本末尾汇总：

| 指标 | 数值 |
|---|---:|
| 平均延迟 | 5524.94 ms |
| 中位数延迟 | 5528.21 ms |
| P99 延迟 | 5568.03 ms |
| 延迟标准差 | 13.89 ms |
| 吞吐量 | 0.18 FPS |
| 平均显存 | 2751.82 MB |
| 峰值显存 | 3272.30 MB |
| 平均功耗 | 6.22 W |
| 最大功耗 | 6.27 W |

## 5. 结果解读

1. **时延稳定性较好**：标准差 `13.89 ms`，且 P99 与中位数差距小，说明单轮推理波动不大。
2. **吞吐量受单次推理时延限制**：当前约 `0.18 FPS`，符合约 `5.5s/次` 的平均延迟量级。
3. **显存占用可控**：平均显存约 `2.75 GB`，瞬时峰值到 `3.27 GB`，峰值高于平均值符合推理过程中的临时张量分配特征。
4. **功耗处于低波动区间**：本次采样均值与峰值接近，表明在该输入与采样策略下功耗曲线较平稳。

## 6. 结论

在 `MAXN_SUPER + jetson_clocks` 满性能状态下，当前算法基线性能为：

- **延迟基线**：约 `5.52 s/iter`
- **吞吐基线**：约 `0.18 FPS`
- **资源基线**：约 `2.75 GB` 平均显存，`3.27 GB` 峰值显存，约 `6.22 W` 平均功耗

可将本报告作为后续模型裁剪、算子优化或推理参数调整（如 timesteps、输入分辨率）前后的对比基准。
