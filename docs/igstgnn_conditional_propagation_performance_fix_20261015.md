# 条件消息传播：静默阶段0与GPU归约修复

2026-10-15。针对服务器停留在 `The number of parameters: 433481` 的现象。本次修复计算实现和可观测性，模型公式、参数布局、batch64、分块4、FP32、确定性模式、损失与选模规则保持原定义；浮点加法顺序允许数值舍入差异。

## 原因与证据

参数日志之后，第一次运行先执行 HISTORY 自身阶段0的完整14批验证，原验证循环没有进度输出。用户进程证据显示 `--worker HISTORY` 仍执行、CPU占用96%；此前GPU利用率97%、显存约8.2/32GB。单凭资源占用不能证明正确，但代码定位与本地复现确认存在严重性能瓶颈。

旧实现对每个receiver块、每个bank执行channel展开的 `scatter_add`。PyTorch 2.3.1在CUDA确定性模式中会进入 `_scatter_via_index_put`，生成坐标索引，再进入排序累加路径；这不是普通原子加法的成本。[官方2.3.1源码](https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L1680-L1704)支持此调用链。

496站点按4个receiver分块，一次核调用有124块；完整模型每batch有20次核调用，每块11个bank。旧实现每batch会执行27280次bank归约，静默验证包含14批。参数数量不能代表这部分计算成本。

## 修复

按已排序receiver边打包成 `[receiver, padded_source, slot]`，对每行source×slot做安全softmax，再用bmm加权求和，移除显式bank scatter归约。padding边严格置零，空bank仍返回0，self rho=1，其他bank rho=3×prior行和；保留共享报告交互、硬支持、bank顺序和原投影。

打包索引为现有拓扑派生的非持久化缓存；旧/新state_dict键和参数相同，严格加载后重建缓存。未请求诊断时跳过极值计算；HISTORY或关闭对应路径时不分配无用的整块零增量。原生gather反向仍可能使用排序累加，不声称整个训练调用图都无scatter。

验证仍调用原engine，不重写指标或掩码。临时包装原val iterator，在阶段开始、每批开始/完成、阶段结束或异常时flush输出JSON，包含arm、stage_zero/epoch_N、completed/total、elapsed_seconds。每批完成标记在原engine已完成反标准化和CPU复制后发出；不增加模型调用、显式同步或RNG操作。

## 本地验证

45项回归全部通过（23 kernel、10整网集成、12 runner，62.937秒），独立复核35项kernel/runner测试亦通过。覆盖独立密集公式及全部梯度、padding/空行、subnormal、硬支持、checkpoint、缓存严格加载、报告回退、时间轴、精确续训及日志异常恢复。另将冻结d36a7c7旧核换回同一完整小模型，开启报告、关闭报告、关闭logit、关闭value、关闭语义五种状态的12槽输出最大差均不超过1.2e-7。

性能输入使用实际发布邻接、合成历史和合成非负图，不读取交通X/Y。硬件为RTX4060 Laptop 8GB、Torch2.8/CUDA12.6；确定性开启、FP32、N496、B64、C32、T12、分块4，收集诊断。CUDA同步计时，旧版本来自本地Git提交d36a7c7，两版使用完全相同参数和输入。

| 范围 | 旧实现 | 新实现 | 输出最大绝对差 | 计时范围 |
|---|---:|---:|---:|---|
| 第一个4-receiver块 | 0.8909秒 | 0.01438秒 | 2.38e-7 | 各预热2次、测5次均值；不含输入准备 |
| 完整单个空间核 | 97.02秒 | 1.484秒 | 3.58e-7 | 各单次；包含图prior展开、检查和全部receiver块 |

分别约62.0倍和65.4倍为这些局部推理范围的观测比值，**不是整网、训练epoch或V100的提速承诺**。单个空间核也不是完整模型，后者每batch调用20次。完整单核GPU峰值分配约4.47GB→1.67GB，为本合成probe的全部当时分配，不是整个训练峰值。未运行正式训练、未评价测试集，预测有效性仍待正式实验。

数值证据：[验证JSON](verification/igstgnn_conditional_propagation_performance_fix_20261015.json)。资源探针：[benchmark_min_conditional_kernel.py](../experiments/chronological/benchmark_min_conditional_kernel.py)。服务器可独立测量单块，输出写入新文件：

```bash
python experiments/chronological/benchmark_min_conditional_kernel.py \
  --adjacency ../data/xtraffic/Contra_Costa/adj_matrix.npy \
  --collect-diagnostics \
  --output experiments/chronological_runs/conditional_kernel_v100_probe.json
```

此探针用于测量而非训练前必做步骤；默认只测单receiver块。`--full-kernel`会额外执行旧版完整慢核，耗时明显更长。

## 服务器使用

若旧命令还在前台，先在原终端按一次Ctrl+C，让launcher及其worker退出。它只保留已经提交到轮次边界的检查点；当前未完成轮次不会保存。等待进程退出后更新代码，使用新目录 `contra_min_conditional_p1_fast_bs64_seed2025` 重跑。完整命令见[运行说明](igstgnn_conditional_propagation_implementation_20261015.md)。

本次修改改变代码指纹；不放宽旧identity、不覆盖旧目录、不声称跨实现精确续训。旧制品继续保留，原MIN源检查点保持只读。

## 流程来源

性能修复采用 optimize-for-gpu 技能的“先冻结基准、验证语义、CUDA同步计时、限制结论范围”流程。软件流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)，2026-10-15核对当前记录为v2。该引用不作为模型有效性证据。
