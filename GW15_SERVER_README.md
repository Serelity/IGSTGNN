# GW15：在事故融合和时空传播前编码历史变化顺序

发布分支 `research/trajectory-encoder-gw15`。本轮直接进入一个内部结构实验，不重做 GW13 查表、GW14 分层或预算匹配，不延长此前任何训练。保留所有旧源码、结果与失败判据。

## 更新并运行

继续使用原 `IGSTGNN-GW9` checkout、`igstgnn` 环境和原认证数据/运行条件。无需找回 GW13/GW14 输出目录，也无需重新训练历史模型。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git fetch origin research/trajectory-encoder-gw15:refs/remotes/origin/research/trajectory-encoder-gw15
if git show-ref --verify --quiet refs/heads/research/trajectory-encoder-gw15; then
  git switch research/trajectory-encoder-gw15
else
  git switch -c research/trajectory-encoder-gw15 origin/research/trajectory-encoder-gw15
fi
git pull --ff-only origin research/trajectory-encoder-gw15
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW11_RUN="$PWD/experiments/chronological_runs/contra_gw11_state_gate_s2025_20261019_164846"
export GW15_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW15_RUN="$PWD/experiments/chronological_runs/contra_gw15_trajectory_encoder_s2025_$(date +%Y%m%d_%H%M%S)"
export GW15_DEVICE="cuda:0"
export GW15_THREADS="3"
unset GW15_RESUME GW15_EVALUATE_ONLY
bash experiments/trajectory_encoder_gw15/run.sh
```

Git 如因本地修改拒绝切换，保存并核对修改，不执行 `reset --hard`。`GW11_RUN` 指向原已完成运行；只用其身份恢复认证来源，不使用其退化的 LAST 作为新起点。真正起点仍是完整 GW9 `SUM_CONTROL BEST`（骨干与已训练 SUM 头）。原绝对来源路径需保留。源码核验失败时不要绕过或放宽回放容差。

入口只运行本轮新增结构/梯度/预算/续跑检查，随后直接训练两组；不重跑旧测试套件。读取和核验已有身份、掩码与缓存是来源完整性检查，没有新增查表拟合或信息诊断阶段。

## 改动与比较

原始 `embedding` 逐时刻映射流量，后续 GRU 和时空模块已经能够感知顺序；本轮不假设它完全丢失顺序。新问题是：显式导数表示在进入事故融合和图传播前，是否使历史变化顺序更容易被现有模型利用？

新增编码器读取 12 个已观测流量产生的 11 个有符号差分，每节点除以 `1+mean(abs(delta))`。候选保留时间顺序；对照对同组差分升序排列，保留其多重集合、净变化和总绝对变化，移除新增编码器可见的原始顺序。**两组完整源模型仍接收真实有序历史。** 排序也改变输入分布，结果不能单独解释为纯粹因果顺序效应。

两组都是 `Linear(11,16)→Tanh→Linear(16,12×32,bias=False)`，各 6336 参数。最后一层初始化全零，输出 `0.1*tanh(...)`，作为 `[batch,12,node,32]` 隐状态残差加入原 embedding 输出。随后执行原 ICSF、动态图、DIF/INH、TIID 和原 SUM 头，不添加最终预测修正头。全部历史在预测起点已可用；不宣称各历史位置的逐时刻因果编码。

| 案例 | 新增编码器输入 | 新增更新 |
|---|---|---:|
| FROZEN_SUM | 无 | 0，仅作固定参照 |
| SORTED_DELTA | 排序后的同一组 11 个差分 | 246 |
| ORDERED_DELTA | 原始时间顺序的 11 个差分，唯一候选 | 246 |

两组新增初始张量相同，阶段零均须逐点、dtype 精确恢复认证 SUM 缓存。完整源模型保持 `eval`、原 `requires_grad` 标志、权重/缓冲区不变，所有源参数排除于 Adam。训练需要通过源模型反向传播，但仅用 `autograd.grad(loss, encoder_parameters)` 计算新增参数梯度，源参数 `.grad` 保持空。**冻结权重不等于把训练前向放入 `no_grad`。** 评价才使用 `no_grad`，精确回放容差不放宽。

保持原归一化、原 MAE、原掩码、batch=48、seed=2025 和全部 12 步。两组固定各 3 轮、每轮 82 批，Adam 学习率 `.002/.001/.001`，沿用 weight decay 与梯度裁剪，同序训练，不加入偏差损失。BEST 包含阶段零；LAST 固定第 3 轮。每个训练批需要两次完整源前向及反向路径，不能像 GW12 那样共享一个脱离梯度的预测；时间和显存以新日志为准，没有本地实测值。

唯一候选 BEST 相对同次 FROZEN_SUM、SORTED_DELTA 各改善至少 0.5%，LAST 相对自身 BEST 退化不超过 1%，且真值 ≥314、步长 7–12、50≤真值<200 三组 MAE 不劣于 FROZEN_SUM，才记为通过本轮结构效用筛查。历史 OWN4 只报告，不是同预算梯度对照。通过不等于创新成立、优于所有历史方法或独立确认；失败停止本轮，保留负结果，不追加轮次或调门槛。

本轮完整开发验证集为同一原生有效掩码，不使用 GW14 的交集掩码，不能直接用 GW14 的 MAE 作为同口径对照。开发验证集已反复参与研究设计。本轮不创建或遍历测试 loader；既有加载器整体反序列化原 object 文件时会包含测试对象。已有单种子和固定源的限制保留。

更详细的历史证据边界、原始接口和有限相关工作核对见 [设计说明](docs/igstgnn_trajectory_encoder_gw15_design.md)。本地仅做静态文本、字节与 Git 审阅，未执行 Python、导入、语法检查、测试或模型；服务器实际日志才可确认数值运行。

## 查看结果与续跑

```bash
cat "$GW15_RUN/report.md"
cat "$GW15_RUN/screening.json"
cat "$GW15_RUN/summary.json"
cat "$GW15_RUN/history.json"
```

完整 BEST/LAST 预测位于 `result_pointer.json` 指向的新 `evaluation_*/best/paired_predictions.npz` 和 `evaluation_*/last/paired_predictions.npz`，对应 metadata 记录来源、选中轮次、前向数量及源码摘要。无需重新手动调用筛查命令。

每个完整轮次把两个编码器、Adam、scheduler、随机状态一起原子提交至 `last.pt`。中断后保留原 `GW15_RUN`，不要重新生成时间戳：

```bash
# 换 shell 时先恢复上面的环境，并把 GW15_RUN 设置为原完整绝对路径。
export GW15_RESUME="1"
unset GW15_EVALUATE_ONLY
bash experiments/trajectory_encoder_gw15/run.sh
```

已提交轮次跳过，未提交轮次从上一完整轮次重放，总提交预算仍为 3 轮。原目录需有 `identity.json` 和 `last.pt`；若初始化尚未提交，保留部分目录并另选新输出。不要删身份标记来冒充续跑。

如果 3 轮已提交，只在最终导出中断：

```bash
unset GW15_RESUME
export GW15_EVALUATE_ONLY="1"
bash experiments/trajectory_encoder_gw15/run.sh
```

查找自己的原 GW15 目录：

```bash
find "$PWD/experiments/chronological_runs" -maxdepth 2 -type f \
  -path '*/contra_gw15_trajectory_encoder_s2025_*/identity.json' -printf '%h\n'
```

不自动选择最新目录，不覆盖现有结果，不安装依赖或自动提交调度器作业。
