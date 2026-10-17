# GW2：复用已有实验，只跑新模型

分支：`research/graph-message-gw2`。以后不在本地执行模型检查，验证全部放服务器。

当前策略：ORIGINAL 和 STATIC 直接读取 GW1 的同预算检查点、best/last 预测归档、训练曲线和历史耗时；不训练、不重新推理。只运行 POST_LOCAL 与 POST_MESSAGE。STATIC 标注为 GW1 历史参考，不能声称它与 GW2 新归约内核逐位相同。源码会校验数据、种子、训练配置、初始化、预算、源代码和缓存 checkpoint/预测哈希；不兼容就明确报错，不自动重跑基线。

## 1. 已在运行的 GW2

若当前日志 arm 为 ORIGINAL/STATIC，按一次 Ctrl+C 停止重复训练。若已经到 POST_LOCAL/POST_MESSAGE，让它继续至退出。完整轮次保存于 last.pt，不删除运行目录。等待当前进程退出后再更新代码。

新增复用入口保留原 `train_original_graph_message.py`、模型和全部训练执行依赖的字节，因此原 a2c5b32 运行的 POST_LOCAL/POST_MESSAGE 可以继续。部分 epoch 未提交时只会按原协议重放该部分所属 epoch。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW2
git fetch origin
git merge --ff-only origin/research/graph-message-gw2
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW2_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW2_RUN=experiments/chronological_runs/contra_original_gw2_s2025
export GW1_RUN=../IGSTGNN-GW1/experiments/chronological_runs/contra_original_gw1_s2025
```

GW1_RUN 必须是之前产生 ORIGINAL 16.28849、STATIC 15.22879 的五轮 GW1 运行目录；如果当时命名不同，按实际路径修改。不能指向最佳轮次28的成熟 original_baseline 或 MIN 运行。源目录应保留 identity.json、两组 last.pt、best 检查点，以及已完成 best/last 评价的 paired_predictions.npz 和 metadata.json。

## 2. 服务器测试与续跑

```bash
python -m unittest tests.test_graph_message_reuse -v
```

测试正常退出后，在已有 GPU 分配中执行：

```bash
bash experiments/chronological/run_original_graph_message.sh --run-dir "$GW2_RUN" --reuse-gw1-run "$GW1_RUN" --data-dir "$GW2_DATA" --epochs 5 --seed 2025 --resume
```

该命令只补两个新动态组未完成的轮次；若两组已完成五轮，不发生任何优化器更新，只生成复用汇总。ORIGINAL/STATIC 在本运行目录已有的部分/完整制品保留，但不用于新比较，也不继续它们。

首次从未启动过 GW2 时，使用同一命令但去掉 `--resume`，运行目录必须不存在。源 GW1 两组必须恰好完成请求预算；若以后需要更长预算，仅补缺失轮次，不从头重跑已经完成的工作。

默认自动在 GW1 目录中寻找匹配的 best/last 预测归档；如有需要，可显式追加：

```bash
# 用实际路径替换下面的示例参数：
# --gw1-best-archive "$GW1_RUN/eval_best_原标签/paired_predictions.npz"
# --gw1-last-archive "$GW1_RUN/eval_last_原标签/paired_predictions.npz"
```

源缓存不存在/证书不匹配时会报错，不会偷偷重算 ORIGINAL/STATIC。不要删除以前的评估目录。

已通过的 GW2 batch48 检查不重复；尚未通过时只检查两个新模型：

```bash
export GW2_CHECK_TAG="$(date +%Y%m%d_%H%M%S)"
python -u experiments/chronological/verify_original_graph_message.py --stress --batch-size 48 --arms POST_LOCAL POST_MESSAGE --data-dir "$GW2_DATA" --output "experiments/chronological_runs/gw2_new_batch48_$GW2_CHECK_TAG.json"
```

没有 Slurm 分配时启动脚本只提交 sbatch，须等作业结束后评价。已有 GPU 分配时前台执行。不要对仍在运行的同一目录重复启动。

## 3. 仅评价新模型，基线直接读缓存

```bash
export GW2_EVAL_TAG="$(date +%Y%m%d_%H%M%S)"
python -u experiments/chronological/evaluate_original_graph_message_reuse.py --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --checkpoint best --output-dir "$GW2_RUN/reuse_best_$GW2_EVAL_TAG"
python -u experiments/chronological/evaluate_original_graph_message_reuse.py --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --checkpoint last --output-dir "$GW2_RUN/reuse_last_$GW2_EVAL_TAG"
cat "$GW2_RUN/reuse_summary.json"
cat "$GW2_RUN/reuse_best_$GW2_EVAL_TAG/report.md"
cat "$GW2_RUN/reuse_best_$GW2_EVAL_TAG/mechanisms.json"
cat "$GW2_RUN/reuse_last_$GW2_EVAL_TAG/report.md"
```

本版七状态：ORIGINAL、STATIC、POST_LOCAL、POST_MESSAGE、POST_LOCAL_UNIFORM、POST_MESSAGE_UNIFORM、POST_MESSAGE_STATIC_ONLY。STATIC_UNIFORM 从未在 GW1 测过，本次不补跑；表中不会伪造这行。metadata.json 明确记录 ORIGINAL/STATIC 新推理次数为0，历史耗时单独标注为旧会话测量。

主比较仍是 POST_MESSAGE vs ORIGINAL，同时保留 vs STATIC、vs POST_LOCAL。五轮仅为结构筛查，不证明收敛优势；测试集不评价。复用证书是归档内部一致性核验，不是历史训练的外部独立认证。
