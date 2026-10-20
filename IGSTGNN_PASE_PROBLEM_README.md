# 原版BEST与PASE BEST：真实预测误差对照

只在同一组837个Contra Costa验证样本上比较已训练好的两套BEST权重，不训练、不追加轮数、不改选模、不分析测试缓存。

## 服务器运行

在 `igstgnn` 环境运行：

```bash
PASE_PROBLEM_REPREDICT=1 bash scripts/run_igstgnn_problem_diagnosis.sh
```

`PASE_PROBLEM_REPREDICT=1`分别从原版完整训练权重/日志和PASE完整100轮运行的验证BEST重新预测一次。
不设置它时优先复用两套已认证的相同BEST验证缓存；缺少某套缓存时只补该模型的验证预测。

默认直接使用用户已经完成的路径：

- 原版权重/日志：`IGSTGNN-code/experiments/igstgnn/Contra_Costa_igstgnn_s2025_20260908_193127/`。
- 已有原版验证：`IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/original_native_val_20261019_223356/`。
- PASE完整运行：`IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025/`。
- 已有PASE BEST验证：上个目录的`eval_best_val_epoch_100/STATE_EXCHANGE/`。

上述路径均相对于`/seu_share/home/huangkai/220243809/paper/IGSTGNN`。可用`PASE_PROBLEM_ROOT`改变父目录。
代码目录、数据目录、原版缓存、PASE运行、PASE缓存、输出、设备可分别用脚本开头的`PASE_PROBLEM_*`变量覆盖。
新预测调用原来运行所在代码目录，避免随意切换模型版本；不为通过认证而放宽源文件或运行环境检查。

## 直接回答的两个问题

1. 原版是否存在区域走势与站点局域变化预测不匹配的真实案例？
2. 对完全相同的真实值、站点、样本和固定覆盖掩码，PASE是否减少站点误差与局域偏离误差？

输出的图含真实值、原版预测、PASE预测以及原版误差阴影；没有合成/平滑流量、虚构事故时刻或拼接重叠窗口。
原版的案例选择完成后才叠加PASE曲线，不按PASE最大提升挑图。PASE更差也照常报告。

## 结果文件

- `report.md`：总体覆盖、分解定义、原版/PASE配对表及解释范围。
- `paired_overall.csv`：两模型在同一诊断子集上的站点、区域平均、局域偏离MAE。
- `per_horizon.csv`：全部12个时距的配对误差。
- `window_regions.csv`：全部合格窗口—区域，无删去反例。
- `case_prediction_error.png`、`.pdf`：真实预测问题图草稿，三站选择依据真实局域变化活动的低/中/高位置。
- `case_plotted_values.csv`：图中每个点可回查。
- `diagnosis.json`：模型与缓存来源、原版模式覆盖、同类记录的PASE改善计数及限制。

固定分区复用GW16的道路/方向/里程≤32站规则。每个区域在12步中使用同一组有限且原始目标>0.001的站点，覆盖至少80%、至少4站。该子集是诊断口径，不与原版native正式验证MAE混算；MAE三项不满足相加关系。

## 结论边界

“OBSERVED”只表示找到了本次明示描述判据的预测模式，不等于已证明普遍模型缺陷。区域变化技能>0只表示比零增量参照好；局域误差大、预测变化较弱也可能含不可预测噪声或合理收缩，不单独证明过平滑。

如果原版没有出现该模式，图明确标为`Pattern NOT OBSERVED`。如果PASE未改善对应误差，配对表和曲线保留负结果。整套PASE改善不独立证明交换算子贡献，后者由已有/后续受控消融回答。

这是在改进已经形成后对重复使用的开发验证集作的回顾诊断，不改写为研究事前发现，不称独立确认或显著性验证。

作者机不运行Python；数值单测由服务器`.sh`先运行，失败会停止后续预测。
