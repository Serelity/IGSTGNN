# v12a：固定 A 的架构机制审计

本轮只实现研究第一步：量化 LayerNorm、ICSF、TIID 和动态图的固定权重响应。
不训练新模型，不据此判断新架构优劣，不自动启动第二步。

## 决定依据与问题

v8 的事件和节点影响识别通过开发门槛，但 v9 的有符号残差专家没有通过收益与
常规对照保护检查。代码显示，单事故 ICSF 的两个 softmax 轴长度为 1；同时，ICSF
会对所有节点最后一帧做 LayerNorm，然后才构造动态图。这些事实需要与预测变化
联系起来，不能直接认定它们导致了之前的失败，也不能把其替代方案预称为“修复”。

固定协议：`experiments/chronological/architecture_mechanism_audit_v12a.json`。
数据和 checkpoint 身份继承已冻结的 v6a 协议，通过文件哈希验证。

## 五条主路径和一次动态图回放

| 路径 | 最后一帧历史 | 图 | TIID |
| --- | --- | --- | --- |
| off | 原始嵌入 H | 根据 H 构图 | 无 |
| norm_only | LN(H) | 根据 LN(H) 构图 | 无 |
| icsf_only | 原始 ICSF 的 LN(H + 事故注入) | 根据 ICSF 历史构图 | 无 |
| tiid_only | LN(H) | 与 norm_only 相同 | 原始报告上下文 |
| full | 原始 ICSF | 原始 ICSF 图 | 原始报告上下文 |
| full_norm_graph | 原始 ICSF | 回放 norm_only 的动态图；静态图共享 | 原始报告上下文 |

每个真实批次的 off 和 full 都与原模型前向直接核对；审计结束重新核验全部参数及
buffer 的哈希不变。原始生产模型和 checkpoint 格式不修改。

回放仅测量“在固定 ICSF 历史、TIID 下改变动态图”的条件敏感性。LayerNorm 改变
历史，也可能改变动态图；`norm_only - off` 包含这些后续变化。不能把所有路径差值
解释为相互独立的模块贡献。另报告 `full - icsf_only - tiid_only + norm_only` 的预测
交互项，不用绝对变化相除生成“贡献百分比”，也不做 MAE 的加法归因。

## 范围与输出

- 只读取 train：完整事故 3,604；匹配事故、C1、C2 各 3,106。读取包级元数据，但不
  读取或哈希验证集数组，不读取测试集。合成探针使用 ICSF 的 CPU 副本计算梯度；
  真实样本只做 inference，无优化器。合成梯度不表示整个模型的梯度。
- C1/C2 用自己的交通 X/Y 和预测时钟，使用配对正样本的报告年龄与距离；保留已有
  共同三元组和唯一的 10-mile 边界兼容规则。
- 每条路径报告全体、候选 H1–H3/H4–H6/H7–H12、非候选早期/晚期和逐 horizon
  的预测变化及描述性 MAE。预测敏感性统计包括全部输出单元；MAE 只用有效 Y。
- NPZ 保存每样本的误差总和、有效计数、预测差值的有符号和/绝对和/最大值及原始
  sample index，可用于后续配对复核。没有进行显著性筛选或以 MAE 排名选新模型。
- 合成探针检查单事故注意力、Q/K 与融合 MLP 的梯度、V 路径的梯度、历史扰动、
  同支持距离扰动和零支持时的 LayerNorm 效应。TIID 仍可通过 K 和距离学习。

同一组训练期样本已用于多轮研究，结果是探索性机制诊断。组件关闭可能造成分布
偏移，不是公平重新训练后的模型比较，也不是事故因果效应或独立确认。

## 服务器运行

在平台分配的 GPU 节点终端执行。沿用已有 `igstgnn` 环境及数据目录。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
conda activate igstgnn
git pull --ff-only origin research/chronological-tiid
bash experiments/chronological/run_architecture_audit.sh start contra_v12a_architecture_01
```

启动器在后台依次执行本轮单元测试、每群体 2 条样本的 `--check`，全部通过后才做
完整训练集审计。默认 CUDA 推理，CPU 负责合成探针和统计。不需要 `/usr/bin/time`。
默认每批 16 条，每 5 批和每群体结束打印进度。完整审计所需服务器时间尚未测量。

```bash
tail -f experiments/chronological_runs/contra_v12a_architecture_01.job/run.log
bash experiments/chronological/run_architecture_audit.sh status contra_v12a_architecture_01
bash experiments/chronological/run_architecture_audit.sh report contra_v12a_architecture_01
```

`Ctrl+C` 只停止 tail。`status` 会核对运行主机，避免把另一节点的 PID 不存在误判为
任务失败。运行记录含 Python 路径、主机、PID、commit 和退出码。

成功输出 `experiments/chronological_runs/contra_v12a_architecture_01/summary.json`，
状态 `ARCHITECTURE_MECHANISM_AUDIT_COMPLETE`。小样本检查仅标记 `ENGINEERING_CHECK_PASS`。
中途失败保留 `.partial/failure.json`、进度和已经完成的群体；硬终止可能来不及写失败
文件。重跑请把运行名改为 `_02`，不覆盖任何已有结果。向用户回传 `report` 输出即可。

若需更小显存占用，可手动调用审计入口传 `--batch-size 8`，使用新输出路径；参数不
改变样本或协议。入口 `--help` 列出数据路径选项。不要在运行期间拉代码。

## 工程验证与交接状态

2026-09-27 在 WSL 创建独立 Conda 环境 `igstgnn-audit`（Python 3.10.21、
PyTorch 2.3.1+cpu、NumPy 1.24.4）。本轮 12 项测试及生产模型、原事故分支、
chronological 数据/训练和时间响应的相关回归，共 52 项全部通过。覆盖真实模型的
路径复现、冻结状态不变、梯度正对照、图固定回放、未来目标与敏感性统计分离、
四群体保存/失败保留，以及工程检查失败时后台流程不继续运行。Bash 语法检查通过。

本地原始训练输入全部哈希通过。四群体分别用预先声明的前 2 条真实样本，在随机
初始化的 496 节点生产架构上完成接口检查；off/full 回放与原生前向的最大差均为零。
这次检查没有计算或报告 MAE，不能替代真实冻结权重检查。

真实冻结 checkpoint 仅在服务器，本地未完成固定 A 的全量复现或 v12a 科学审计。
待服务器结果回传后再决定是否设计节点门消融。已有 v11b 未提交文件及研究日志
修改保留原状，v12a 的协议与本轮研究记录独立提交。
