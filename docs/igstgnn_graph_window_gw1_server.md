# GW1 独立分支：服务器更新与四组五轮试跑

仓库：<https://github.com/Serelity/IGSTGNN>。
新分支：`research/graph-window-gw1`。
分支起点：原远端 `research/chronological-tiid` 的 `aea9f6f0aaba26c02d90210724fe9ad577f26eb5`。
GW1 单独提交到新分支；不推送、合并或重置原分支。服务器也使用独立 Git worktree，保留原开发目录和已有实验。

## 1. 首次建立独立目录

在现有仓库运行以下命令。`../IGSTGNN-GW1` 应为尚不存在的新目录；不需要复制数据、删除结果或 stash 原目录改动。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git fetch origin
git worktree add --track -b research/graph-window-gw1 ../IGSTGNN-GW1 origin/research/graph-window-gw1
cd ../IGSTGNN-GW1
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
git log -1 --oneline
```

默认数据目录仍为共享父目录下的 `data/xtraffic/Contra_Costa`。若服务器部署在其他位置，对 Python 检查/评价及训练命令添加 `--data-dir /实际数据目录`；不在此重新拟合归一化或生成不同划分。

后续更新在独立目录执行：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW1
git pull --ff-only origin research/graph-window-gw1
```

有活动训练时先完成它再更新其源码。每个新终端重新激活 `igstgnn` 并设置上述两个环境变量；脚本也会默认从自身路径定位 GW1 仓库。

## 2. 在已分配 GPU 的终端做 batch48 检查

可先用四行真实数据做小批等价性和梯度检查：

```bash
"$PYTHON_BIN" -u experiments/chronological/verify_original_graph_window.py --check --output experiments/chronological_runs/gw1_check_s2025.json
```

再严格使用 batch48，分别测 ORIGINAL 和 GRAPH_CONTENT 的原尺度 MAE 前向/反向：

```bash
"$PYTHON_BIN" -u experiments/chronological/verify_original_graph_window.py --stress --batch-size 48 --output experiments/chronological_runs/gw1_batch48_s2025.json
```

结果应为 `GW1_VERIFICATION_PASS`；两组 `backward.*.batch_size` 均为 48。这个检查没有优化器更新，不训练、不评价测试集，也不会在 OOM 后自动减小 batch。输出文件必须是新文件，重复检查时另换文件名。

## 3. 四组各五轮试跑

```bash
bash experiments/chronological/run_original_graph_window.sh --run-dir experiments/chronological_runs/contra_original_gw1_s2025 --epochs 5 --seed 2025
```

同一运行依次完成 `ORIGINAL`、`STATIC`、`SHARED_DYNAMIC`、`GRAPH_CONTENT` 四组，每组从相同原模型初始化开始、batch48、各完成 5 轮。`--epochs 5` 不是四组总计五轮，也不是加载历史最佳检查点后微调。
无 Slurm 分配时脚本自行提交；已在 GPU 作业内则直接执行。批次进度和每组阶段信息会打印，Slurm 日志为该 GW1 目录的 `original-gw1-作业号.out`。

进程已退出或作业已结束后，使用相同环境和目录续训：

```bash
bash experiments/chronological/run_original_graph_window.sh --run-dir experiments/chronological_runs/contra_original_gw1_s2025 --epochs 5 --seed 2025 --resume
```

累计预算仍是每组五轮，不会再追加五轮；保留已有目录和锁文件。若原进程还在运行，同一目录会拒绝重复启动。

## 4. 完成后的指标和诊断

```bash
cat experiments/chronological_runs/contra_original_gw1_s2025/summary.json
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_graph_window.py --run-dir experiments/chronological_runs/contra_original_gw1_s2025 --checkpoint best --output-dir experiments/chronological_runs/contra_original_gw1_s2025/eval_best
"$PYTHON_BIN" -u experiments/chronological/evaluate_original_graph_window.py --run-dir experiments/chronological_runs/contra_original_gw1_s2025 --checkpoint last --output-dir experiments/chronological_runs/contra_original_gw1_s2025/eval_last
```

读取两个 `report.md` 与 `mechanisms.json` 判断结果。主比较为完整组相对原 IGSTGNN；同时查看两种简化结构，以及同检查点 UNIFORM/STATIC_ONLY 两种干预。
负收益、低流量/MAPE 恶化、best/last 差异和成本均保留；五轮不代表收敛结论。

## 5. 发布内容和已有工程证据

新增内容限于 GW1 模型、训练/验证/评价入口、测试、文档及小型工程证据；没有上传数据、检查点或完整本地实验目录。
复用的原训练辅助、冻结九文件与 v0.3 数值统计函数已与分支父提交核对一致，不包含本地其他未发布模型改动。
实现定义见 [GW1 方案](igstgnn_graph_window_gw1.md)。

- 原开发记录：[development](verification/graph_window_gw1_development_20261017.json)。其中 `pushed: false` 和本地运行路径记录的是发布前状态。
- 随本分支提供的证据：[小批检查](verification/graph_window_gw1_evidence_20261017/check.json)、[batch48](verification/graph_window_gw1_evidence_20261017/stress48.json)、[精确续训](verification/graph_window_gw1_evidence_20261017/exact_resume.json)。
- [发布核对记录](verification/graph_window_gw1_publication_20261017.json) 区分原开发快照与当前发布文件哈希，并记录独立工作区回归测试和启动脚本检查。

工程验证不构成精度超过原模型的证据；正式五轮运行与服务器 V100 的显存/耗时须以上述实际输出为准。
