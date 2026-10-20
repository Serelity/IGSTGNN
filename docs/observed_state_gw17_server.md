# GW17 服务器运行与续跑

本地只编写和静态检查；所有Python、单测、工程前向和数值训练只在服务器执行。
使用既有 `igstgnn` conda 环境，默认gpu05、cuda:0、3线程，不需要新增依赖。
GitHub固定提交的无Git下载命令见本次交付回复；源码放到新的GW17目录，不覆盖旧GW16代码。

在新下载目录运行：

```bash
conda activate igstgnn
bash experiments/observed_state_gw17/run.sh
```

脚本会依次执行：新增单元测试 → 两组各3轮真实小样本工程检查 →
工程BEST/LAST归档和复算 → 两组各100轮正式训练 → 正式BEST/LAST验证 → 固定筛查 → 打印结果。
只有OBSERVATION_CONTROL和INNOVATION_STATE是本轮新训练组；GW16只读取验证缓存。
测试集不参与本轮训练、选模或筛查，没有test入口。

默认数据：
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa`。

默认GW16参照：
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025/eval_best_val_epoch_100/STATE_EXCHANGE`。

默认GW17结果在**新下载代码目录**下
`experiments/chronological_runs/contra_gw17_observed_state_s2025`；工程结果名字追加 `_engineering`。
脚本打印所有实际路径，并保存新代码目录的 `gw17_paths.env`，不依赖此前shell的GW16变量。
如果数据或参照确实移动，可设 `GW17_DATA`、`GW17_REFERENCE`、`GW17_RUN`；已有运行恢复时必须沿用原有输入和代码。

## 中断后

仍在同一GW17下载目录、同一conda环境执行原命令即可：

```bash
bash experiments/observed_state_gw17/run.sh
```

脚本自动识别identity并添加resume，不需要查最新模型文件。
已完成的组跳过；进行中的组恢复完整Adam、scheduler、RNG与累计轮数。
单轮未提交的部分重新训练该轮；不会重置优化器、延长100轮或覆盖旧实验。
初始化中断且缺identity的目录会拒绝自动恢复并打印路径；保留它后显式设置新的GW17_RUN。

如需要先分段运行，轮数表示累计目标，不是追加：

```bash
GW17_EPOCHS=20 bash experiments/observed_state_gw17/run.sh
GW17_EPOCHS=100 bash experiments/observed_state_gw17/run.sh
```

部分预算只打印训练summary，不执行正式筛查；最终目标仍为每组100轮。

## 查看结果

在新代码目录执行：

```bash
source ./gw17_paths.env
cat "$GW17_RUN/summary.json"
cat "$GW17_RUN/eval_best_val_epoch_100/report.md"
cat "$GW17_RUN/eval_last_val_epoch_100/report.md"
cat "$GW17_RUN/screening/screening.json"
```

每组 `history.json` 包含未来val_MAE、偏差、区域MAE、历史更新前MAE、辅助损失和完整耗时。
历史MAE不是未来val_MAE，未参与BEST选择；目标是判断观测偏差校正是否改善未来预测。
报告中GW16_REFERENCE的正式验证pooled MAE来自已认证缓存，不能与原文测试表混算。
`candidate_for_confirmation` 仅表示预先规定的开发筛查通过，不等于新独立测试或显著性结论。

设计、损失、归因边界与失败处理见 [设计说明](observed_state_gw17_design.md)。
