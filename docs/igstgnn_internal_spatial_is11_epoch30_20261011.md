# IS1.1 三十轮结果：最佳检查点尚不支持受限交换优势

日期：2026-10-11。作业1173577，gpu05，session_20261011_203528_4fc79a；主run仍为contra_is11_20261011_163218_259a3f。用户回传四组epoch11—30共80条日志，四组均完成累计30轮，汇总状态IS11_PAIRED_STAGE_COMPLETE；本阶段没有组在30轮前早停。

证据仅为用户回传stdout。[派生JSON](verification/igstgnn_is11_server_epoch30_20261011.json)拼接此前40条记录，独立复核全部120条的轮次、running best、末5轮及汇总差值。原始服务器history、checkpoint和逐窗口数组尚未下载；算术复核不等于服务器原始产物验收。历史见[十轮分析](igstgnn_internal_spatial_is11_epoch10_20261011.md)。

## 1. B2稳定性先行

| B2诊断 | epoch10训练 | epoch30训练 | epoch10验证 | epoch30验证 |
|---|---:|---:|---:|---:|
| 潜状态裁剪比例 | 7.0996% | 4.8955% | 7.0920% | 4.9086% |
| 最大截断绝对量 | 0.895278 | 0.902148 | 0.863502 | 0.906058 |

与第10轮相比，裁剪比例下降，但个别最大截断没有消失，验证最大值略升。比例按算子支持位置的潜状态元素、窗口、调用及子步累计，不能说4.9%的道路或样本被裁剪。只有少数阶段端点的汇总，不能声称全程单调改善或稳定性问题已解决。

目前没有NaN/梯度错误或提前中断的回传证据；这也不认证B2状态路径梯度健康。B2裁剪是既定对照的一部分，不因看到排名就单独修改它的步长或读出。B3/B2的差距仍包含有界更新等多种算子差异，不能直接全归为真实交通物理。

## 2. 按预先约定的选模指标报告结果

主指标为全496站、12时距的验证horizon-macro MAE，越低越好。最佳值选自各组前30轮；第30轮为共同轮次，不能用第30轮替代各组最佳值。

| 组 | 空间机制 | 前30轮最佳MAE | 最佳轮次 | 第30轮MAE |
|---|---|---:|---:|---:|
| B0 | 原ICSF | 23.958116 | 28 | 24.994919 |
| B1 | 原ICSF＋强度门控 | **23.864951** | 30 | **23.864951** |
| B2 | 报告状态＋普通交换 | 23.992344 | 28 | 24.128287 |
| B3 | 报告状态＋受限交换 | 24.208890 | 27 | 24.642378 |

收益为control_MAE−candidate_MAE，正值表示候选更好。

| 比较 | 最佳MAE收益 | 相对control最佳值改善 | 第30轮MAE收益 |
|---|---:|---:|---:|
| B1相对B0 | +0.093166 | +0.3889% | +1.129969 |
| B2相对B1 | −0.127393 | −0.5338% | −0.263336 |
| B3相对B2 | −0.216546 | −0.9026% | −0.514091 |
| B3相对B0 | −0.250773 | −1.0467% | +0.352541 |
| B3相对B1 | −0.343939 | −1.4412% | −0.777427 |

按既定最佳检查点标准，本阶段没有支持B3优于原模型或同信息普通交换。第10轮B3/B2的最佳值收益为+0.346990，本阶段变为−0.216546，说明早期优势没有保持到当前选模结果。不能只取第30轮B3优于B0的结果，忽略B0第28轮更好的检查点。

B1最佳值仍领先，但其相对B0的改善从第10轮3.54%缩小到0.39%。这只说明当前单种子的差值，不能据此宣布稳定提升、统计等价或显著优于其他组。

## 3. 走势提供补充，不能改变选模规则

| 组 | 末5轮平均MAE（26—30） | 末10轮平均MAE（21—30） | epoch11—30平均秒/轮 |
|---|---:|---:|---:|
| B0 | 24.453470 | 25.070701 | 58.65 |
| B1 | 24.573907 | 25.139415 | 58.90 |
| B2 | **24.353777** | 24.734703 | 91.36 |
| B3 | 24.557527 | **24.672278** | 100.52 |

末10轮平均值是B3最好，末5轮是B2最好，最佳检查点是B1最好。B3在21—22轮较少出现B0/B1那样的大幅反弹，但第28轮仍升到25.2270；不能称为持续稳定领先。B3/B2末10轮平均收益仅0.062425，末5轮平均收益则为−0.203751，结论对观察区间敏感。

这些平均值可描述优化过程；不改用其中有利的区间选胜者，也不把相邻epoch当作独立重复做显著性检验。不同阶段的单个MAE差值亦不能作为事故机制的贡献量。

B3单轮仍比B0慢约71.4%、比B2慢约10.0%；当前最佳检查点尚无对应收益，新增计算成本应保留在后续评价中。

## 4. 这轮定位了什么，尚未回答什么

已回答：在当前内部替换、输入、种子与前30轮预算下，“报告状态＋受限交换”没有在既定选模指标上胜过原模块、简单门控或同信息普通交换。IS1.1的完整预测创新目前尚未成立，不能沿用第10轮的有利结论。

尚未回答：事故报告是否真正改变了新容量路径、内部道路交换是否承担有效信号、硬预算是否长期处于饱和或几乎不激活，以及改善/损失是否集中在报告关联节点。当前stdout没有这些数据，不能断定失败来自容量估计、约束过强、报告被忽略或梯度受阻。

以下只是需要诊断区分的假设，不是本次日志已经证实的原因：

- 报告状态更新很小：先核对采用强度、posterior与prior差值及报告状态梯度，判断报告路径是否学到可用信号。
- 容量发生变化而内部交换响应很小：核对提议量、实际内部通量、预算激活与开放边交换；边界主导不能证明内部道路传播有效。
- 受限交换经常削弱有用表示：核对各层预算、状态/通量及读出梯度；与B2的同信息比较结合，不能只凭全网MAE归因为“物理过强”。
- 报告关联位置改善而其他位置退化：回读既定分组和道路/时距指标，判断全网结果是否掩盖区域差异；不以局部收益替代全网主指标。

四组均保留原TIID；B2/B3同时替换原ICSF直接注入，B3/B0还改变候选道路结构。因此当前不能把结果扩大为“事故信息无用”或“物理模型不适合这个数据”。当前潜交换也不代表已认证的真实交通容量或守恒。

## 5. 下一阶段安排

先回读已保存history里的分组与机制摘要，不需要新训练或新的推理。优先看各组最佳轮次与共同第30轮，避免只挑一个有利检查点。诊断重点为报告采用/更新、容量相对prior的变化、实际内部与开放边通量、预算激活/超限和新增路径梯度。首个训练batch的梯度只代表该batch，不能据其一次为零就判定整个路径失活。

在服务器仓库目录运行以下只读命令，打印截至30轮的最佳及第30轮摘要；不导入Torch、不读checkpoint/test、不写训练产物：

~~~bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("experiments/chronological_runs/contra_is11_20261011_163218_259a3f")
for arm in ("B0", "B1", "B2", "B3"):
    history = json.loads((root / arm / "history.json").read_text(encoding="utf-8"))
    history = [row for row in history if row["epoch"] <= 30]
    assert [row["epoch"] for row in history] == list(range(1, 31)), arm
    best = min(history, key=lambda row: row["validation_mae_macro"])
    for label, row in (("best_to30", best), ("epoch30", history[-1])):
        metrics = row["validation"]["metrics"]
        out = {
            "arm": arm, "snapshot": label, "epoch": row["epoch"],
            "metrics": {
                name: {
                    "mae_macro": value.get("mae_macro"),
                    "valid_count_per_horizon": value.get("valid_count_per_horizon", value.get("counts"))
                }
                for name, value in metrics.items()
            },
            "all_nodes_mae_per_horizon": metrics["all_nodes"]["per_horizon_mae"]
        }
        for phase in ("train", "validation"):
            m = row[phase]["mechanisms"]
            if m is None:
                continue
            records = m["rows"]
            out[phase] = {
                "report_state": m["report_state"],
                "state_clip": m["b2_stability"],
                "sums": {
                    key: sum(r[key] for r in records)
                    for key in ("internal_flux_sum", "inlet_flux_sum", "exit_flux_sum",
                                "proposal_sum", "capacity_limited_count", "sending_limited_count",
                                "receiving_limited_count", "positions", "edge_positions")
                },
                "maxima": {
                    key: max(r[key] for r in records)
                    for key in ("capacity_violation_max", "sending_violation_max",
                                "receiving_violation_max", "float64_balance_abs_max")
                }
            }
        gradients = row["train"]["first_batch_gradient_l1"]
        out["first_batch_gradients"] = {
            name: value for name, value in gradients.items()
            if name.startswith(("report_spatial_state.", "icsf_module.strength_gate."))
            or "readout.weight:" in name
        }
        print(json.dumps(out, ensure_ascii=False))
PY
~~~

通量和激活计数跨调用/子步累计，不是车辆流量、事故数或真实分钟；发送/接收计数与容量计数的分母也不能混用。B2记录的是假想限制，B3才是实际限制。B3的state_clip字段沿用公共记录键，不代表它通过裁剪保持有界。此摘要不替代完整逐层分析，但可先定位是否存在报告更新、预算参与和梯度断点。

预测实验保留同一run与四组，下一执行阶段目标为累计50轮，保持原上限100、patience20、学习率计划和输入。第38、46轮后的学习率下降已经属于冻结协议；当前各组最佳都出现在27—30轮，尚不能认定已完成收敛。50只是便于Slurm执行与回读的阶段端点，不替代既定上限或早停规则。不得根据是否逆转排名临时修改学习率或延长单个组。

~~~bash
bash experiments/chronological/run_internal_spatial.sh \
  --run-dir experiments/chronological_runs/contra_is11_20261011_163218_259a3f \
  --epochs 50
~~~

每组计划新增20轮，保留原早停；全部完成的轮内时间粗估103分钟。已有脚本可直接运行，无需更新训练代码，Git不进入Bash。

本阶段不直接扩展多种子、不新增模块，也不单独淘汰B1或B2。先完成机制核对及当前冻结训练流程；若后续出现值得验证的收益，再按既定条件扩展空间报告OFF从头重训练和配对种子。若完整训练仍无收益，再依据具体断点另立下一版内部改造，保留本轮失败结果与对照身份。test继续封存。
