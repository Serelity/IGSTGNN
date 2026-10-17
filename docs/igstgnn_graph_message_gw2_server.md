# GW2 服务器运行

仓库：https://github.com/Serelity/IGSTGNN  
分支：`research/graph-message-gw2`  
新目录：`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW2`

从已存在的 GW1 checkout 创建独立 GW2 worktree，保留 GW1 及原开发目录。以下为首次建立新目录的命令，不用于覆盖已存在的 GW2 工作目录。检查、训练、评价全部放服务器执行；本次发布未运行本地模型检查。

## 1. 获取代码

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW1
git fetch origin
git worktree add --detach ../IGSTGNN-GW2 origin/research/graph-message-gw2
cd ../IGSTGNN-GW2
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW2_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW2_RUN=experiments/chronological_runs/contra_original_gw2_s2025
export GW2_TAG="$(date +%Y%m%d_%H%M%S)"
git log -1 --oneline
```

此 worktree 使用 detached HEAD 固定当前代码。正式运行时不要更新执行源码，否则原有运行的严格续训身份会拒绝接续。

## 2. 在已分配 GPU 的终端检查

当前 `gpu14/gpu05` 终端需有可用 GPU 分配；登录节点不要直接运行 CUDA 检查。若尚无 GPU 分配，按集群原有流程申请后，再进入上面的 GW2 目录设置变量。

```bash
python -c 'import torch; print("CUDA:",torch.cuda.is_available()); print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"); assert torch.cuda.is_available()'
```

先执行服务器回归测试和小批量恢复检查，再执行真实 batch48 前后向。输出文件名含新时间标签，不覆盖旧结果；失败时不缩小批量冒充通过。

```bash
python -m unittest tests.test_original_graph_message tests.test_original_graph_message_runner tests.test_original_graph_message_evaluation -v
python -u experiments/chronological/verify_original_graph_message.py --check --data-dir "$GW2_DATA" --output "experiments/chronological_runs/gw2_check_$GW2_TAG.json"
python -u experiments/chronological/verify_original_graph_message.py --stress --batch-size 48 --data-dir "$GW2_DATA" --output "experiments/chronological_runs/gw2_batch48_$GW2_TAG.json"
```

最后一条正常退出后确认状态，只有 `GW2_VERIFICATION_PASS` 才进入试跑：

```bash
python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["status"]); print({a:{k:v.get(k) for k in ("batch_size","forward_backward_seconds","peak_allocated_bytes")} for a,v in d.get("backward",{}).items()}); assert d["status"]=="GW2_VERIFICATION_PASS"' "experiments/chronological_runs/gw2_batch48_$GW2_TAG.json"
```

## 3. 四组各五轮

ORIGINAL、STATIC、POST_LOCAL、POST_MESSAGE 均从相同原始参数初始化，沿用原损失、Adam、学习率、课程与 batch48。`--epochs 5` 是每组总完成五轮。首次训练目录必须不存在。

```bash
bash experiments/chronological/run_original_graph_message.sh --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --epochs 5 --seed 2025
```

已有 Slurm 作业分配时，上述命令前台执行，返回后才进行下一步。没有 `SLURM_JOB_ID` 时启动脚本通过 sbatch 提交，返回只表示提交成功；须等该作业完成后再评价。此时日志在 checkout 的 `original-gw2-<jobid>.out`。

被中断后，在新 GPU 分配中进入同一目录、恢复同样环境变量，再执行：

```bash
bash experiments/chronological/run_original_graph_message.sh --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --epochs 5 --seed 2025 --resume
```

不要对仍运行的同一目录重复启动；运行锁会拒绝并发。GW1 权重不能作为 GW2 的续训检查点。

## 4. 训练完成后分别评估 best 和 last

```bash
export GW2_EVAL_TAG="$(date +%Y%m%d_%H%M%S)"
python -u experiments/chronological/evaluate_original_graph_message.py --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --checkpoint best --output-dir "$GW2_RUN/eval_best_$GW2_EVAL_TAG"
python -u experiments/chronological/evaluate_original_graph_message.py --run-dir "$GW2_RUN" --data-dir "$GW2_DATA" --checkpoint last --output-dir "$GW2_RUN/eval_last_$GW2_EVAL_TAG"
cat "$GW2_RUN/summary.json"
cat "$GW2_RUN/eval_best_$GW2_EVAL_TAG/report.md"
cat "$GW2_RUN/eval_best_$GW2_EVAL_TAG/mechanisms.json"
cat "$GW2_RUN/eval_last_$GW2_EVAL_TAG/report.md"
```

需要对比 POST_MESSAGE vs ORIGINAL、vs STATIC、vs POST_LOCAL，不能只展示最优一组。五轮用于结构筛查，不证明完整预算收敛优势或超过论文测试结果；本流程不评价测试集。
