# B3R-MIN：切回 IGSTGNN 作者发布数据

日期：2026-10-13。此文替代精简版原先默认的 v8 开发数据运行说明。

## 当前默认入口

`run_b3_r_minimal.sh` 和 `train_b3_r_minimal.py` 默认使用 `paper` 数据源，
进入 `train_b3_r_minimal_paper.py`。未找到作者发布文件就报错，不会回退到 v8。

自动依次查找仓库下 `data/xtraffic/Contra_Costa`、仓库同级
`../data/xtraffic/Contra_Costa`。也可用 `--data-dir` 或
`IGSTGNN_PAPER_DATA_DIR` 指定作者数据目录。

## 数据与训练约定

- 读取本地作者发布的 `incident_all.npy`，固定 SHA256，原文件不修改。
- 按作者 `prepare_splits.py` 的数组顺序与取整规则切分：3,909 / 837 / 839 条。
  发布包实际为 5,585 条，与论文 Table 1 的 5,587 条差两条，不能擅自补造。
- 直接使用发布 `x_data` 的 12×496×3：流量、日内时钟、星期。
  目标直接使用发布 `y_data` 的 12 个流量步，不重新抽窗，不加入两个空槽。
- 数据已经归一化，沿用 `incident_stats.npz`，不再次归一化、不拟合新统计。
- 原图卷积使用发布 `adj_matrix.npy` 及原 `doubletransition` 处理。
  P 的时钟取最后历史槽，取消 v8 专用的“加两个槽”。
- 精简架构保留 P、边界入口/出口、受限潜在交换与五层主干；不重新加入 TIID、D
  或事故编码。候选有向交换图仍是模型附加结构，不宣称它是原论文邻接矩阵。
- 复用现有 `IGSTGNN_Engine` 的原批次构造、逆归一化、掩码与评价规则；
  batch 48、Adam、学习率计划、早停 20、上限 100 轮。
  采用原课程学习 `warm_epoch=30, cl_epoch=3`，并保存课程进度。
- 默认只训练与验证。原 all 文件是对象数组，反序列化不可避免包含测试载荷，
  审计明确记录 `test_payload_deserialized=true`，不声称测试内容从未加载。
  不计算测试指标；最终需要时显式增加 `--evaluate-test`，使用最佳验证权重评价。

原发布顺序不等于我们独立认证过绝对日历或跨集合无重叠；这里忠实保留发布协议，
不把旧版本缺陷静默修复成另一套数据。这也不是已经复现论文表格分数。

## 服务器运行

先同步本次代码；启动脚本不含 Git，不申请 `--mem`。

工程检查只用前四条训练和验证样本，不用于模型排名：

```bash
bash experiments/chronological/run_b3_r_minimal.sh --check
```

从头运行至 10 轮，使用独立新目录：

```bash
bash experiments/chronological/run_b3_r_minimal.sh \
  --run-dir experiments/chronological_runs/contra_b3r_min_paper_v1_seed2025 \
  --epochs 10
```

在同一目录续跑至总计 30 轮：

```bash
bash experiments/chronological/run_b3_r_minimal.sh \
  --run-dir experiments/chronological_runs/contra_b3r_min_paper_v1_seed2025 \
  --epochs 30
```

已有 Slurm 作业内直接运行；登录节点会自动提交。Python 直接入口相同：

```bash
python -u experiments/chronological/train_b3_r_minimal.py \
  --run-dir experiments/chronological_runs/contra_b3r_min_paper_v1_seed2025 \
  --epochs 10
```

若自动查找未命中，追加 `--data-dir /实际路径/Contra_Costa`，目录内应有原发布的
六个文件。SHA256 不同会报错，不能通过改 JSON 指纹混入另一版数据。

## 检查结果与续跑

输出保存 `identity.json`、`data_audit.json`、初始化证书和各会话环境记录；
`B3R_MIN/last.pt` 原子保存模型、最佳权重、优化器、学习率、课程步数、
样本排列及 NumPy/Torch/CUDA 随机状态。重复请求已完成轮数不会重复训练。
中断以最后完整轮次恢复；伴随 JSON 可从已提交检查点恢复。

数据源、文件指纹、时钟规则、训练协议和源码进入独立身份。
**原 v8、B3R 或旧精简实验目录不能在这里续跑，必须从头训练。**

原 v8 程序仅作为历史入口保留，需显式 `--data-source chronological`。
该模式与新主实验不能混排；已有面向 v8 的跨模型评估器不应用于本次结果。
新结果先看本目录的 `summary.json`、`history.json` 与显式测试报告。

完整 IGSTGNN 基线仍需在相同发布数据和训练预算下对照，才能判断精简版效果。
本次只切换用户指定的精简模型，不自动启动其他组。

## Batch 64 运行选项

保留冻结协议的默认 batch 48，新增 `--batch-size 64` 同时设置训练和验证批次。
数据文件、切分、归一化、模型结构和初始学习率均不变；实际 batch 保存进运行身份
和 summary.json。课程学习仍按轮数换算更新步数。`--check` 固定使用 batch 2，
不能用该检查推断 batch 64 的峰值显存。

先在新目录完成一轮完整训练，观察峰值显存与耗时：

```bash
bash experiments/chronological/run_b3_r_minimal.sh \
  --run-dir experiments/chronological_runs/contra_b3r_min_paper_bs64_seed2025 \
  --batch-size 64 --epochs 1
```

一轮正常完成后，用同一参数续跑到总计 10 轮：

```bash
bash experiments/chronological/run_b3_r_minimal.sh \
  --run-dir experiments/chronological_runs/contra_b3r_min_paper_bs64_seed2025 \
  --batch-size 64 --epochs 10
```

续跑时须始终保留 `--batch-size 64`。已有 batch 48 实验不直接改 batch 续跑；
本次源码更新也会改变运行身份，旧实验若需精确续跑，应保留原入口与源码版本。
batch 改变会影响每轮更新次数和优化轨迹；之后的原版 IGSTGNN 对照也应使用 64。
是否更快、是否放得下，仍需服务器完整批次实测。

## 本地验证

22 项针对精简模型、原发布数据适配、batch 参数、续跑与既有评估兼容性的测试通过。
原发布包全部训练/验证样本通过形状、时钟和有限值检查；六个文件指纹匹配。
496 站、前四条训练/验证、CPU batch 2 的一轮工程检查通过，参数 435,856。
该小批检查不是预测效果结果，也未验证服务器 V100 全 batch 的运行。

工程结果目录：
`experiments/chronological_runs/contra_b3r_min_paper_check_20261013_025726_36c5de/`。
