# GW4 服务器检查、适配训练与结果读取

源码分支：`research/receiver-residual-gw4`。本轮所有模型运行、语法检查、单元测试与训练均留在服务器执行；本文中的命令尚未执行，不能据此视为验证通过。分支获取命令要求该分支已发布，当前发布状态以交付说明为准。

## 1. 独立代码目录和既有结果

分支发布后，在服务器创建独立目录；不要切换或覆盖正在使用的 GW2/GW3 目录：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
git clone --single-branch --branch research/receiver-residual-gw4 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW4
cd IGSTGNN-GW4
```

已有这个 GW4 checkout 时使用 `git fetch origin` 和 `git merge --ff-only origin/research/receiver-residual-gw4`。有活动运行时先保留其源码身份，不在训练中间更新文件。

在已分配 GPU 的终端设置：

```bash
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW4_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW2_RUN=/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW2/experiments/chronological_runs/contra_original_gw2_s2025
export GW4_RUN=experiments/chronological_runs/contra_original_gw4_frozen_s2025
export GW4_TAG="$(date +%Y%m%d_%H%M%S)"
```

`GW2_RUN` 必须是已完成并生成 `reuse_best_*` / `reuse_last_*` 认证预测的真实 GW2 目录；其保存的 GW1 引用也必须仍可读。程序自动查找身份、轮次、检查点和生产者源码均匹配的已有缓存，找不到就停止，不重训或重新推理旧组。

若自动查找失败，可在检查和训练命令上明确增加 `--gw2-best-eval /已有的/reuse_best_时间标签`、`--gw2-last-eval /已有的/reuse_last_时间标签`。这里传入评估目录，不是 npz 文件。不要用报告重分析目录或 GW3 评估目录代替 GW2 生产者证书。

## 2. 先检查，再适配

先运行服务器单元测试和完整 batch48 工程检查：

```bash
"$PYTHON_BIN" -m unittest tests.test_receiver_residual_workflow -v &&
"$PYTHON_BIN" -u experiments/chronological/verify_original_receiver_residual.py \
  --gw2-run "$GW2_RUN" \
  --data-dir "$GW4_DATA" \
  --batch-size 48 \
  --device cuda:0 \
  --output "experiments/chronological_runs/gw4_check_$GW4_TAG.json"
```

单元测试覆盖完整轮次恢复、第一轮中途失败后回放、冻结张量篡改拒绝、负收益保留、BEST/LAST 同张量缓存与报告掩码。工程检查的微型合成模型覆盖非均匀旧评分下的零残差恢复、旧模块 eval、梯度分阶段到达、有限差分、全递推 BASE 恢复、干预异常恢复和优化器/RNG 恢复。

随后每个新组只在一批真实数据上做全 12 时距 native MAE 前向/反向，不执行真实数据 optimizer.step，不生成旧基线验证预测。合成夹具内部会执行少量优化器更新，JSON 单独记录，不能把整个检查说成零优化器更新。

成功后 JSON 应显示 `GW4_VERIFICATION_PASS`、`full_native_backward_checked=true`。仅合成检查不能冒充真实 batch48 通过；OOM 不自动减小批量。

确认检查完成后，执行三个新组，各最多五个适配轮：

```bash
bash experiments/chronological/run_original_receiver_residual.sh \
  --gw2-run "$GW2_RUN" \
  --data-dir "$GW4_DATA" \
  --run-dir "$GW4_RUN" \
  --epochs 5 --seed 2025
```

在 Slurm GPU 作业内直接运行；不在作业内时 launcher 提交 sbatch。若在已明确分配 GPU 的非 Slurm 环境运行，可按既有方式设置 `ORIGINAL_IGSTGNN_DIRECT=1`。

三个组分别为：

| 组别 | 新增评分 | 活跃参数 |
|---|---|---:|
| STATIC_RESIDUAL | 来源×槽位偏置 | 15 |
| MESSAGE_RESIDUAL | 偏置 + 消息均值查询与消息键的点积 | 1,055 |
| RECEIVER_RESIDUAL | 偏置 + 接收状态查询与消息键的点积 | 1,055 |

参数量按当前 C=32、K=3、来源=5、隐藏=16 计算；实际计数由服务器写入 identity。三个组注册量一致，但 STATIC 的查询与键冻结且不参与前向，不能称三个组活跃容量都相同。

## 3. 预算、冻结与断点续跑

源固定为旧 GW2 POST_MESSAGE 的 BEST。新 Adam 仅接收新增活跃参数，学习率、L2、裁剪、损失等数值沿用既有配方；旧模块保持 eval。新调度器和课程从适配第零步开始，前五轮仍按原 warm 配方使用完整 12 时距。这不是原优化器的连续续训，也不是与旧 GW2 等预算的从头训练比较。

每轮结束检查旧张量摘要，保存模型、优化器、调度器、课程、RNG、训练顺序与完整历史。恢复时仍传目标总适配轮数，不把 `--epochs` 当作本次追加轮数：

```bash
bash experiments/chronological/run_original_receiver_residual.sh \
  --gw2-run "$GW2_RUN" \
  --data-dir "$GW4_DATA" \
  --run-dir "$GW4_RUN" \
  --resume --epochs 5 --seed 2025
```

完成的组直接跳过；未完成一轮从最后完整轮回放。预算上限为 5，不能降低已保存的目标，也不能把旧五轮重复加入新预算。`--only RECEIVER_RESIDUAL` 可暂时只推进一组，但配对评估要求三个组最终完成共同预算。

BEST 只在新增适配轮中选择，旧锚点单列，不借助第零步把新组的失败藏起来。`GW4_ADAPTATION_COMPLETE` 表示完成预算，不表示性能改进。

## 4. BEST/LAST 配对评估

```bash
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_receiver_residual.py \
  --run-dir "$GW4_RUN" --data-dir "$GW4_DATA" \
  --checkpoint best --output-dir "$GW4_RUN/eval_best_$GW4_TAG" &&
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_receiver_residual.py \
  --run-dir "$GW4_RUN" --data-dir "$GW4_DATA" \
  --checkpoint last --output-dir "$GW4_RUN/eval_last_$GW4_TAG"
```

两份评估都以源 GW2 BEST 为锚点。ORIGINAL/GW2_BASE 仅取认证历史预测；不调用 BASE 干预重跑旧模型。若一个新组的 BEST 与 LAST 模型张量一致，后执行的评估复用该组全部既有预测及干预状态。不同张量的新模型需正常生成新预测。

新增同检查点干预是 `STATIC_ONLY`（仅关闭新增点积，保留新增偏置和旧 GW2 评分）、`QUERY_SELF`（仅替换新增接收查询）。它们不替代独立训练组。全部评估只使用验证集。

读取：

```bash
cat "$GW4_RUN/summary.json"
cat "$GW4_RUN/eval_best_$GW4_TAG/report.md"
cat "$GW4_RUN/eval_last_$GW4_TAG/report.md"
```

重点检查 RECEIVER_RESIDUAL 是否同时超过 MESSAGE_RESIDUAL、STATIC_RESIDUAL 和 GW2_BASE。收益定义为“对照误差−候选误差”。若只优于旧 ORIGINAL，尚不足以保留新模块。

`analysis.json`、`cases.csv`、`comparisons.csv`、`horizons.csv`、`groups.csv` 提供指标；`gw2_hard_windows_and_shift.csv` 另列 GW2 固定困难窗口收益和平均预测移动；`mechanisms.json` 保存调用阶段、来源、权重极值、残差和查询/键范数；`metadata.json` 保存身份、角色、哈希与实际新增推理次数。

既有 `groups.csv` 的固定困难组沿用 ORIGINAL 最差 10% 窗口；GW4 主报告另用 GW2_BASE 固定最差 10%，两者分别标识，不能混作同一组。真实未来流量分组只作事后诊断。

只重分析已经生成的 GW4 npz 时使用新输出目录：

```bash
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_receiver_residual.py \
  --archive "$GW4_RUN/eval_best_$GW4_TAG/paired_predictions.npz" \
  --output-dir "$GW4_RUN/reanalysis_$GW4_TAG"
```

该方式没有推理或训练，也不独立重新认证原生产者，报告会明确标注。

## 5. 当前证据边界

本轮源码准备与服务器验证分开记录；当前没有新 MAE、显存、速度或通过率结果。5 轮筛查后仍须根据配对结果决定是否值得做完整预算、多种子和独立确认。新增来源门、事故字段、时延层或效用伪标签均未加入本轮。
