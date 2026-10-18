# GW5 服务器运行

分支：`research/history-anchor-gw5`。使用独立目录，复用已完成 GW4 三组及其 GW2 引用。以下检查、训练和评估尚未在本会话执行。

## 获取源码和设置路径

分支发布后，首次获取：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
git clone --single-branch --branch research/history-anchor-gw5 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW5
cd IGSTGNN-GW5
```

已有 GW5 checkout 时在该目录执行 `git fetch origin`、`git merge --ff-only origin/research/history-anchor-gw5`。活动运行中保持源码版本固定。

在已分配一张 GPU 的同一终端设置：

```bash
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW5_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW4_RUN=/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW4/experiments/chronological_runs/contra_original_gw4_frozen_s2025
export GW5_RUN=experiments/chronological_runs/contra_original_gw5_history_s2025
export GW5_TAG="$(date +%Y%m%d_%H%M%S)"
```

`GW4_RUN` 是上次已经完成五轮的目录。它保存的 GW2/GW1 引用必须仍可读。本轮不再次执行旧实验。

程序自动发现已认证 GW4 BEST/LAST 评估；若需要指定，可在检查和训练命令增加 `--gw4-best-eval /已有的/eval_best_时间标签`、`--gw4-last-eval /已有的/eval_last_时间标签`。传评估目录，不是 npz 文件。缺失时停止，不自动重建缓存。

## 服务器检查

```bash
"$PYTHON_BIN" -m unittest tests.test_history_anchor tests.test_history_anchor_workflow -v &&
"$PYTHON_BIN" -u experiments/chronological/verify_original_history_anchor.py \
  --gw4-run "$GW4_RUN" \
  --data-dir "$GW5_DATA" \
  --batch-size 48 --device cuda:0 \
  --output "experiments/chronological_runs/gw5_check_$GW5_TAG.json"
```

单元检查覆盖零残差/GW2恢复、FREE/GW4恢复、偏置保留、历史梯度、有限差分、跨batch重置、异常清空、完整轮与首轮中断恢复、旧证书、不同策略缓存隔离、负收益和困难集合。

真实 batch48 检查仅对四个新候选各执行一次完整12时距 MAE 前向/反向；没有真实数据 optimizer.step，没有旧控制组前向。合成单元测试会在微型夹具上更新参数，不能将全部单元测试说成零更新。

成功时应出现 `GW5_VERIFICATION_PASS`、`full_native_backward_checked=true`。OOM 不自动改小批量。

## 四组五轮训练

检查通过后，在当前已分配 GPU 的作业内直接运行：

```bash
"$PYTHON_BIN" -u experiments/chronological/train_original_history_anchor.py \
  --gw4-run "$GW4_RUN" \
  --data-dir "$GW5_DATA" \
  --run-dir "$GW5_RUN" \
  --device cuda:0 --epochs 5 --seed 2025
```

四个新组为 HISTORY_FIXED、HISTORY_ANCHORED、PHASE_SHRINK、BIAS_SHRINK。全部从原 GW2 和零残差开始，各五轮，与现有 GW4 的适配预算匹配。没有从已训练 GW4 继续加五轮。

若需 launcher 自动提交 Slurm，使用 `bash experiments/chronological/run_original_history_anchor.sh` 加上相同参数。在 Slurm 作业内它直接执行，不在作业内则提交 sbatch；随后须等待训练作业完成再评估。

训练中断后，以相同路径恢复：

```bash
"$PYTHON_BIN" -u experiments/chronological/train_original_history_anchor.py \
  --gw4-run "$GW4_RUN" \
  --data-dir "$GW5_DATA" \
  --run-dir "$GW5_RUN" \
  --device cuda:0 --resume --epochs 5 --seed 2025
```

`--epochs 5` 是目标总适配轮数。完成的组跳过，未完成一轮从最后完整轮回放。1–4轮可保存阶段结果，旧五轮 GW4 不作为这些短预算的配对精度对照；完整评估要求四组均完成五轮。旧锚点不参与新组 BEST 选择。

## BEST/LAST 配对评估与结果读取

```bash
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_history_anchor.py \
  --run-dir "$GW5_RUN" --data-dir "$GW5_DATA" \
  --checkpoint best --output-dir "$GW5_RUN/eval_best_$GW5_TAG" &&
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_history_anchor.py \
  --run-dir "$GW5_RUN" --data-dir "$GW5_DATA" \
  --checkpoint last --output-dir "$GW5_RUN/eval_last_$GW5_TAG"
```

旧 ORIGINAL、GW2、GW4 三组只读取认证缓存。新模型相同张量且相同策略时复用已有预测；改变策略即使张量相同也不能复用。FREE/ANCHOR_ROLL 是同检查点敏感性，不是重训删除，也不是原 GW4 的替代品。

读取并贴回：

```bash
cat "$GW5_RUN/summary.json"
cat "$GW5_RUN/eval_best_$GW5_TAG/report.md"
cat "$GW5_RUN/eval_last_$GW5_TAG/report.md"
```

`GW5_ADAPTATION_COMPLETE` 表示完成预算。`GW5_EVALUATION_COMPLETE` 表示生成评估。`analysis.json` 的 `history_anchor_screening` 独立列出预先声明的四项开发条件，不能把完成状态当作性能通过。

主要文件：`horizons.csv` 最终12时距；`groups.csv` 原有道路/流量/支持区；`fixed_hard_and_shift.csv` 固定 GW2/GW4 两套困难集合和预测偏移；`mechanisms.json` 各调用/来源的自由残差、历史偏离及有效残差；`metadata.json` 来源、角色、生产者、哈希和实际调用数。

只读重分析已有 GW5 预测：

```bash
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_history_anchor.py \
  --archive "$GW5_RUN/eval_best_$GW5_TAG/paired_predictions.npz" \
  --output-dir "$GW5_RUN/reanalysis_$GW5_TAG"
```

重分析不运行模型，也不独立重新认证原生产者，报告会标明。所有结果限于探索性开发验证；没有测试集评价或已确认泛化结论。
