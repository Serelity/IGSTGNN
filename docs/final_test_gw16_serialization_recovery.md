# GW16 已冻结测试的无损缓存恢复

用户服务器已通过35项测试，并完成ORIGINAL_IGSTGNN全部18批推理、写盘及基本checksum检查。
随后在 `validate_cached` 的严格指标/runtime复核处失败；STATE_EXCHANGE尚未开始推理。

代码中有明确的布局处理遗漏：

- 冻结原版 `model.py:156-157` 从连续 `[B,N,H]` 转置输出 `[B,H,N,1]`。
- 已验证的原版val入口在 `.numpy()` 后调用 `.copy()`，显式转为C布局。
- 新test入口省略这次拷贝，并在非C连续预测上算FP32指标。
- NPZ保存数值和C/F标记，不保存这种任意步长。重读后值相同，FP32归约顺序可能改变。

JSON中的runtime和metric_contract字段均为稳定标量/列表/字典，没有tuple类型往返问题。
现场是否完全由布局造成，将由恢复后对原metadata的**严格相等复算**确认，不放宽容差。
NumPy格式只记录C/F顺序，可参照[官方格式说明](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)。

## 运行

将本次固定提交下载到一个独立新目录。在该目录执行：

```bash
conda activate igstgnn
bash experiments/final_test_gw16_recovery/run.sh
```

用户已确认的旧下载目录为 `IGSTGNN-GW16_TEST_e5dac19`，正式输出仍为
`IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025_final_test_locked`。
脚本使用上述实际绝对路径，以及用户日志中的freeze ID：
`5a9d88244cbe9d0f5e2c6c1fd42a271a920ece32eda5012befe8bd21eb30af06`。

恢复入口通过原来的绝对目录加载原e5dac19代码；原代码文件、freeze、NPZ和metadata都保持原字节。
不会覆盖原目录、创建替代freeze、重新选BEST、重训或重新推理已完成的原版模型。
新包装器仅改变缓存读取后的内存布局；新恢复代码的SHA及执行记录另存 `serialization_recovery_*.json`。

如果直接重读指标已经严格一致，就原样复用；否则仅对ORIGINAL预测做一次固定布局恢复：
连续 `[B,N,H,1]` 再转置回 `[B,H,N,1]`。检查逻辑顺序下全部float32位不变，
并要求全部native指标与原metadata严格相等。STATE_EXCHANGE不允许使用这个恢复分支。
runtime、数据、权重、样本轴、共同掩码指标仍受原检查约束；不一致会输出具体字段并停止。

日志出现 `cache layout restored; saved metrics reproduced exactly` 才确认本次布局诊断。
之后应出现原版 `zero new model forwards`，再执行STATE_EXCHANGE的18批测试。
最后仍输出 `GW16_FINAL_TEST_COMPLETE` 和原路径的 `report.md`。
中断后重复同一恢复命令，已完成模型缓存继续复用。

## 检查范围

新增8项服务器回归测试：合成非均匀数据、float32逐位保持（含负零）、
既有C布局不动、固定恢复后仍不匹配就拒绝、STATE拒绝恢复、
原文件SHA不变、真实冻结native指标保存/重读/恢复的严格复算。
其中确定性布局回调保证恢复分支在不同CPU实现上都被覆盖；真实指标不强制所有CPU都出现末位差异。

开发机仅做静态检查，不执行Python或数值实验。服务器先跑这8项，再恢复既有测试。
这次为缓存表示修复，模型、冻结选择、目标、公式与测试预算保持一致，不构成新的研究候选。
