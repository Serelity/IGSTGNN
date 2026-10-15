# 原始 IGSTGNN 模块筛查与新基线训练

2026-10-15。用户授权先审计原 IGSTGNN 内部模块，随后允许重新训练完整原版基线。此次开发提供一个完整基线训练入口和十二状态固定检查点筛查入口；没有更改模型结构，没有为十二种状态分别重训，没有调参或评价测试集。

## 原版身份与数据

冻结 `558ab01` 的模型、原数据字段解析、传感器解析、指标和训练 engine；通过独立命名空间加载，不调用当前已修改的 `src/models/igstgnn.py`。每次运行验证冻结源码逐字节 SHA256，严格加载完整455220个注册参数。原ICSF跨节点softmax、原TIID四个潜预测块和原gap=3输出头均保留。

读取并验证作者发布的 `incident_all.npy` 与五个配套文件指纹。沿原数组顺序取3909个训练窗口、837个验证窗口，batch48，验证尾批21；不建立测试loader。原object数组整体反序列化会加载包含测试样本的payload，但不检查其目标、不预测、不选模、不评价。这里复用作者发布顺序，不声称独立核验绝对日历时间顺序。

新训练沿用原宽度、五层、Adam学习率0.002、weight decay 1e-5、原scheduler、warm_epoch30、cl_epoch3、最多100轮和patience20。外层训练循环省略旧engine末尾自动测试，训练batch和验证沿用原engine。确定性算法开启、TF32关闭，与历史启动器的运行时设置存在区别；不声称逐位复现历史训练轨迹。代码身份、运行时、数据、样本置换、优化器、scheduler、课程进度、最佳轮次与文件指纹均保存。

## 筛查定义

十二状态为FULL，两类forecast读出OFF，ICSF_ONLY、NORM_ONLY、TIID_ONLY_ON_NORM、RAW_ALL_REPORT_OFF，三类图bank置零，以及GRU输入恒等替代、attention query恒等替代。定义见[审计设计](igstgnn_original_module_value_audit_20261015.md)。

先完整回放FULL并核对最佳验证MAE，失败则阻止所有干预。每种状态使用同一检查点、原batch与mask；两次完整验证重复要求预测逐位一致；每batch结束恢复FULL，核对参数/buffer、输入、RNG及hooks。新训练来源还核对原训练锁、已提交状态、历史最佳选模和更新预算。

**效用定义为“干预误差−FULL误差”。它衡量当前完整模型对某路径的条件依赖，不是删除后重训的结构收益。** 两类forecast OFF保留backcast/残差；GRU与attention替代仍执行原运算再替换输出，可能引起分布偏移。多种干预路径重叠，不能相加为贡献百分比。

输出全网MAE/RMSE/MAPE、逐时距、道路、事后真值流量、固定FULL困难窗口、正负抵消、响应方向、配对预测及恢复证书。真值分层只用于诊断。

## 本地旧最佳检查点实测

源ZIP：`Contra_Costa_igstgnn_s2025_20260908_193127_20260909090248.zip`；最佳epoch27；checkpoint SHA256为`77e221cbfd2e395fc049e44de38d71cbd88e7d2de5e6a1fb0cb0f2542f792f73`。原生FP32指标回放MAE **13.865386009**，历史日志为**13.865386**；保持原18个验证batch。历史启动的完整源码执行证书缺失，回放通过不补造历史签名。

以下为837窗口、12时距的有效点等权pooled MAE；统计采用FP64，因此FULL与原生FP32回放末位稍有差异。

| 状态 | MAE | 干预误差−FULL |
|---|---:|---:|
| FULL | 13.865385650 | — |
| DIF_FORECAST_READOUT_OFF | 77.957766682 | +64.092381032 |
| INH_FORECAST_READOUT_OFF | 52.620121202 | +38.754735552 |
| GRU_TRANSITION_IDENTITY | 41.018980346 | +27.153594696 |
| ATTENTION_QUERY_IDENTITY | 39.171126247 | +25.305740597 |
| DYN_BANKS_ZERO | 20.802338109 | +6.936952459 |
| PRE_BANKS_ZERO | 18.246456745 | +4.381071095 |
| STA_BANKS_ZERO | 15.433070523 | +1.567684873 |
| RAW_ALL_REPORT_OFF | 15.142907563 | +1.277521913 |
| TIID_ONLY_ON_NORM | 13.865409725 | +0.000024075 |
| NORM_ONLY | 13.865409586 | +0.000023936 |
| ICSF_ONLY | 13.865385492 | -0.000000158 |

DIF/INH预测读出、GRU、attention与三类图均有明显条件依赖，优先研究其内部机制和更好的替代结构，不能机械删除。三图屏蔽的本检查点退化顺序为动态、预定义、学习静态，但没有配对重训练/多种子稳定性证明。

事故路径需把归一化分开：RAW_ALL_REPORT_OFF走原生无报告分支，同时绕过ICSF中的LN；其+1.28不能全部归因事故内容。保留LN后，报告注入和TIID增量的净影响很小，TIID关闭甚至仅为1.6e-7量级MAE变化，不构成稳定负收益证据。NORM_ONLY与FULL的预测绝对变化均值约0.00362，正负效用高度抵消；不能只根据小净均值声称完全未读取事故。这里仍不能否定作者的重训练消融结果。

`STLocalizedConv.bn`在原forward中未读取，共320个闲置仿射注册参数。`DistanceFunction.bn`仍活跃；attention的out_proj虽无module调用，参数通过函数式计算参与，因此“零hook调用”不能普遍推出参数闲置。

完整数值报告见[保存的筛查报告](verification/igstgnn_original_module_screen_report_20261015.md)，来源/检查摘要见[验证记录](verification/igstgnn_original_module_screen_20261015.json)。本轮完整数值运行发生在最终入口保护和profiler全部接入前；随后仅离线重汇总相同预测并修正道路方向标签，原制品保持不变。最终入口另通过新训练检查点的小规模GPU衔接验证，不能把两项写成“最终代码已完整重跑837窗口”。

## 推理耗时与模块内部耗时

本机RTX4060 Laptop，FP32、3线程；不是服务器V100训练耗时。十二状态按batch/pass轮换顺序，先warmup；H2D、加载、逆标准化、统计和哈希不计入前向计时。FULL的34次batch48调用均值**0.517635秒**，2次尾批21均值**0.241891秒**；每次837窗口完整推理均值**9.041691秒**，按样本分摊**10.802498毫秒**。后者是批吞吐口径，不是单样本延迟。

另对同检查点首个原batch48做一次warmup及两次被动FULL模块profiling，预测与归档逐位一致。完整CUDA前向均值614.070毫秒，含大量探针事件开销，不能与上面的普通完整推理直接做提速比较。

| 同类模块的各独立实例 | 每完整前向inclusive累计毫秒 | 每前向调用 |
|---|---:|---:|
| 五层DifBlock | 411.965 | 5 |
| 五层InhBlock | 127.724 | 5 |
| STLocalizedConv | 392.586 | 20 |
| GRUCell | 15.728 | 45 |
| MultiheadAttention | 79.690 | 20 |
| DynamicGraphConstructor | 20.785 | 1 |
| ICSF | 3.913 | 1 |
| TIID | 3.372 | 1 |

表内父子时间重叠，例如卷积已包含在DifBlock中，attention已包含在InhBlock中，不能把所有行相加。相同类的独立实例在本表内汇总，不把父子类互相相加。这里只定位当前前向的主要开销；屏蔽操作仍执行原模块，不能据其差异宣称删除后的速度，更不能外推训练/反向提速。

## 服务器执行

更新到本次提交后，在仓库根目录逐条运行。默认脚本在登录节点提交Slurm任务；在已有GPU分配中直接使用当前任务，保持原batch48，不传MIN参数。

先做4个训练/验证窗口的工程检查，验证环境与源码指纹：

```bash
bash experiments/chronological/run_original_igstgnn_baseline.sh --run-dir experiments/chronological_runs/contra_original_baseline_check_s2025 --check --epochs 1
```

通过后正式重训一次完整原版，默认最多100轮、patience20：

```bash
bash experiments/chronological/run_original_igstgnn_baseline.sh --run-dir experiments/chronological_runs/contra_original_baseline_s2025
```

训练完成后，复用该目录选出的单个最佳检查点筛查；不训练十二个状态：

```bash
bash experiments/chronological/run_original_igstgnn_module_ablation.sh --source-run experiments/chronological_runs/contra_original_baseline_s2025
```

若正式训练中断，停止旧进程后明确续训；`--epochs`表示总目标轮次：

```bash
bash experiments/chronological/run_original_igstgnn_baseline.sh --run-dir experiments/chronological_runs/contra_original_baseline_s2025 --resume --epochs 100
```

旧进程持锁时不会另开训练。达到目标或early-stop的续训为no-op。续训要求源码、数据、设备/运行时、配置身份一致，升级代码后不能强行续同一目录；已有阶段文件保留。

PyTorch 2.3的`Counter`检查点兼容问题已通过外置驱动修复，两个bash入口自动使用。原身份所核验的训练/评估/runtime/source文件保持`ac277ea`字节，已有目录可按原命令续训，不需要放宽身份检查；见[兼容修复说明](igstgnn_original_checkpoint_compat_20261015.md)。

已有原版结果也可用`--archive /absolute/result.zip`，或`--checkpoint /absolute/final_model_s2025.pt --training-log /absolute/record.log`直接筛查。两个来源互斥，不能把MIN权重当原版。

主要制品为`summary.json`、`report.md`、`paired_predictions.npz`、`cases.csv`、`comparisons.csv`、`timings.csv`、`module_timings.csv`与JSON明细。自动输出到新目录并打印路径；失败保存failure，不覆盖原报告。

## 完成的验证

- 56项CPU测试通过，覆盖源码身份、数据边界、干预覆盖/恢复、真值mask、指标守恒、计时pass样本核对、profiler、断点恢复及伪选模证书拒绝。
- 本机GPU新训练1轮→续到2轮，共4次工程优化更新；重复续到2轮no-op；该最佳权重接十二状态筛查和模块profiling成功。
- 旧最佳完整837窗口回放、十二状态两次重复、输入/权重/RNG保持与最终FULL逐位恢复均通过；本筛查优化更新0、测试评价0。
- Bash两个入口通过`bash -n`。本轮未在服务器执行正式重新训练，未给出删除后重训或多种子结论。
