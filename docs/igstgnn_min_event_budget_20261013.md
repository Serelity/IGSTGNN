# MIN-I1.0：事故条件有向边预算的开发与运行

日期：2026-10-13。首版在作者发布的 Contra Costa 数据上比较 A / L / S，目标是验证事故位置与语义条件能否改善 MIN。预测收益待服务器完整训练确认。

## 三组与输入

| arm | 模型 | 参数量 | 新增条件 |
| --- | --- | ---: | --- |
| A | MIN | 435856 | 原历史流量、时钟、原图与候选有向图 |
| L | MIN-LOC | 437508 | 发布事故—站点关系 D |
| S | MIN-SEM | 438016 | 相同 D，加类型和描述类别 |

L/S 共用相同尺寸、相同初始化的预算适配器，S 增加 508 个语义编码参数。L 的 16 维语义槽输入为零，相应权重没有该输入的任务梯度，因此不宣称两组有效参数容量完全匹配。S−L 衡量语义条件化版本的训练增量；L−A 是位置条件与新增结构的联合效果；S−A 是完整改造的实际效果。最终仍需与同配置原 IGSTGNN 比较。

沿用作者发布 `incident_all.npy` 的 3909 / 837 / 839 划分、12 历史 / 12 目标、原标准化、原掩码与时钟 offset 0；不重建 v8、不插额外空槽。默认 batch **64**，`--check` 固定 batch 2。发布文件和候选图均核对固定指纹、站点顺序。

Type/Description 使用各自类别 embedding，合法类别 0 保留真实含义；缺失另用 UNK 和有效标记。报告存在、关系可用、节点关联分别编码。`Incident Time`、`durations`、Holiday、常量 position 不进入新模型。D 按原 float32 转换后计算支持，不当公里或真实上下游距离。缺少报告关联时适配器退化为同权重 MIN 运算；这不等于独立训练 A 的权重。

当前是发布事件属性条件下的离线比较，没有认证每条描述在预测截点前已形成。数据对象数组加载会反序列化测试载荷，但默认不检查测试事件、计算测试指标或按测试选择结构；数据审计诚实记录这一区别。

## 模块接口

P 先生成原基础边预算 C0。共享小适配器 A 在每个局部交换子步读取有向边两端的隐藏特征、当前交换状态、C0、端点关系，以及可选事件编码，生成 a=tanh(A)。只对关联内部边更新：

```text
C = C0 + mask * a * where(a >= 0, 1 - C0, C0 - 0.001)
```

每子步都从 C0 重算，不累加此前修正；有效 C 同时进入原提议网络和硬边上限，再经过原发送/接收预算。边界入口/出口、原图卷积、读出和预测头保持原组织。适配器末层置零，初始完整预测与 A 完全一致，L/S 共同适配器权重逐张量相同。

四个预算通道仍为无量纲隐藏量。层和两次子步不是物理分钟；这不是未来容量轨迹，也不识别真实道路容量或事故因果影响。预算变化可能被非瓶颈或收发约束抵消，不能单独视为机制成功。

## 服务器更新

Git 只在更新时单独执行；Bash 启动脚本不含 Git。进入现有仓库：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

工程检查先运行针对性测试，再进行真实 496 站、前四条训练/验证的一轮三组检查：

```bash
bash experiments/chronological/run_min_event_budget.sh --check
```

预期初始化为 `MIN_I_INITIALIZATION_PAIR_PASS`，最终为 `MIN_I_PAIRED_STAGE_COMPLETE`。小样本分数不能用于模型排名，也不能推断完整 batch 64 的峰值显存。

先在独立新目录跑一轮完整数据：

```bash
bash experiments/chronological/run_min_event_budget.sh \
  --run-dir experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --epochs 1
```

完整首轮正常后，同目录续跑至累计 10 轮：

```bash
bash experiments/chronological/run_min_event_budget.sh \
  --run-dir experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --epochs 10
```

`--epochs` 是累计目标。中断后重复原命令，已完成的组/轮次不会重复更新。仅补 S 时保留原三组身份，用 `--only`：

```bash
bash experiments/chronological/run_min_event_budget.sh \
  --run-dir experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --only S --epochs 10
```

登录节点会自动提交 Slurm；已有作业分配中直接执行。脚本申请 1 GPU / 3 CPU / 6 小时，不使用 `--mem`；使用固定 igstgnn Python，因此 `(base)` 提示符也能运行。可通过 `MIN_I_REPO_DIR`、`PYTHON_BIN` 覆盖实际路径。Python 直接入口：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/train_min_event_budget.py \
  --run-dir experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --epochs 10
```

若完整 batch 64 显存不足，三组统一改 batch 并使用独立新目录，例如 `--batch-size 48`。不能改正在续跑的实验 batch。当前没有服务器新模型全 batch 实测，不能预先保证显存或速度。

## 保存、恢复与评价

每组保存 `last.pt`、`history.json`、`summary.json`；检查点包含当前及最佳模型、Adam、scheduler、课程进度、等待早停计数、样本排列、NumPy/Torch/CUDA RNG。检查点原子提交，JSON 中断损坏会留原字节备份并由合法检查点修复；合法但冲突的历史或摘要拒绝覆盖。

源码、数据、图、协议、初始状态、batch 和组选取进入运行身份；`--only`、累计 epoch 目标不改变身份。GPU SKU 名称只记环境，不作为恢复拒绝条件。旧 MIN、旧 B3-R、v8 目录不能用本入口续跑。

三组在独立子进程中顺序运行；父进程在启动 worker 前释放完整数据载荷。父锁与每组 worker 锁分别防重叠。中断会终止并等待子进程，若 launcher 被强杀而 worker 存活，其组锁仍保护正在写入的检查点。失败返回码和日志保存在 `session_*/`；非零退出后可按完整轮次恢复。

原课程/早停不变：第 30 轮末监督长度由 12 切到 1，之后每 3 轮增加一步；早停耐心 20。摘要明确区分 `EARLY_STOPPED` 与普通轮次暂停。达到早停后仅增加 `--epochs` 不会重新启动优化，不静默清除 wait。

阶段对比使用共同完成前缀，列共同轮次和共同前缀最佳验证 MAE、末五轮；每轮保存样本排列哈希并检查三组一致。10 轮用于观察，不单凭早期排序淘汰。

验证同时记录预算修正、局部实际通量响应、容量/收发约束违反最大值和收支误差。通量响应是**相同当前 ON 隐状态下，将当前子步 C 替换为 C0 并重算各自提议**的局部诊断，不是完整模型 OFF 轨迹；分母含 batch、内部窗口、边及潜通道，不能当独立道路样本数。其后按道路、时距和困难窗口的完整收益归因属于后续分析入口。

测试需显式 `--evaluate-test`，只使用最佳验证权重，控制台打印完整各时距与平均结果并保存各组 `test_metrics.json`。若只读测试，可把 `--epochs` 设为已完成轮数，避免额外训练。默认只训练/验证。

## 验收记录

本地验收结果见 `docs/verification/igstgnn_min_event_budget_20261013.json`。覆盖显式缺失信息、原 X/Y 与归一化一致、零初始化及报告 OFF 恒等、完整预测任务梯度、共享参数、当前状态重算、非瓶颈零响应反例、预算有界与收支、逐组精确续跑、合法旁档冲突拒绝、进程中断和锁保护。

设计思想、文献及研究边界见随附 `igstgnn_min_event_budget_plan_20261013.md`。开发仅实现首轮 A/L/S 训练与局部机制诊断，未预先声明事故语义有益、状态条件化优于静态更新或新版本超过原 IGSTGNN。
