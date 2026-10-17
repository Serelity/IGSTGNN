# GW3 服务器运行说明

当前开发版本未执行本地检查，必须先在服务器通过以下检查。服务器使用独立分支 `research/receiver-match-gw3`。不要在仍运行任务的GW1/GW2目录切换分支。

首次获取代码，在现有IGSTGNN父目录创建独立GW3目录：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
git clone --depth 1 --single-branch --branch research/receiver-match-gw3 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW3
cd IGSTGNN-GW3
```

若GW3目录已在该分支，只更新它，不再clone：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW3
git fetch origin
git merge --ff-only origin/research/receiver-match-gw3
```

## 准备

在独立的 `IGSTGNN-GW3` 目录获得本版本代码后，进入该目录并激活环境。以下绝对路径按现有服务器布局；GW2_RUN必须指向本次已完成五轮的GW2目录。

```bash
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export ORIGINAL_IGSTGNN_REPO_DIR="$PWD"
export GW3_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW2_RUN=/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW2/experiments/chronological_runs/contra_original_gw2_s2025
export GW3_RUN=experiments/chronological_runs/contra_original_gw3_s2025
export GW3_TAG="$(date +%Y%m%d_%H%M%S)"
```

## 服务器检查

在已有GPU资源分配内执行；CUDA不可用时先解决资源/环境，不改成CPU跑batch48，也不复用失败验证文件名。

```bash
python -m unittest tests.test_original_receiver_match tests.test_receiver_match_workflow -v
python -u experiments/chronological/verify_original_receiver_match.py --check --data-dir "$GW3_DATA" --output "experiments/chronological_runs/gw3_check_$GW3_TAG.json"
python -u experiments/chronological/verify_original_receiver_match.py --stress --batch-size 48 --data-dir "$GW3_DATA" --output "experiments/chronological_runs/gw3_batch48_$GW3_TAG.json"
```

两个验证文件应为 `GW3_VERIFICATION_PASS`；stress中三组实际batch_size均须48。检查不做优化器更新；单元测试中会在小型合成模型上检验梯度学习和断点续跑。

可选的真实发布数据两轮工程试跑只检查管线，不替代五轮正式试跑；它使用4行训练/验证、batch2，不能复用全量历史指标：

```bash
bash experiments/chronological/run_original_receiver_match.sh --check --run-dir "experiments/chronological_runs/gw3_engineering_$GW3_TAG" --data-dir "$GW3_DATA" --epochs 2
```

## 三个新增组各五轮

```bash
bash experiments/chronological/run_original_receiver_match.sh --run-dir "$GW3_RUN" --data-dir "$GW3_DATA" --gw2-run "$GW2_RUN" --epochs 5 --seed 2025
```

仅训练FIXED_UNIFORM、MESSAGE_SELF、RECEIVER_MATCH。自动查找现有GW2认证best/last评估；不重训或重新推理ORIGINAL、STATIC、POST_MESSAGE。现有GW2目录必须保留其 `_reuse/reference.json` 相邻目录与原GW1来源，不能只搬一个report.md。

自动匹配失败时，用已有评估的实际目录显式指定，保留完整名字：

```bash
bash experiments/chronological/run_original_receiver_match.sh --run-dir "$GW3_RUN" --data-dir "$GW3_DATA" --gw2-run "$GW2_RUN" --gw2-best-eval "$GW2_BEST_EVAL" --gw2-last-eval "$GW2_LAST_EVAL" --epochs 5 --seed 2025 --resume
```

上条的两个评估环境变量需先设置为实际已有目录；只在第一次自动匹配已创建GW3运行目录后加`--resume`，全新目录不要加。

被kill后在同一个运行目录、同一代码和相同配置上加`--resume`。epochs仍写总数5，完整轮次跳过，未提交的半轮从上一完整轮重放。不要删除active.lock来绕过活动进程。

```bash
bash experiments/chronological/run_original_receiver_match.sh --run-dir "$GW3_RUN" --data-dir "$GW3_DATA" --gw2-run "$GW2_RUN" --epochs 5 --seed 2025 --resume
```

无SLURM_JOB_ID时启动器提交sbatch，有资源分配时前台运行。任务完成后再评估；不要训练仍在运行时启动评估。

## 评估与读取

```bash
python -u experiments/chronological/evaluate_original_receiver_match.py --run-dir "$GW3_RUN" --data-dir "$GW3_DATA" --checkpoint best --output-dir "$GW3_RUN/eval_best_$GW3_TAG"
python -u experiments/chronological/evaluate_original_receiver_match.py --run-dir "$GW3_RUN" --data-dir "$GW3_DATA" --checkpoint last --output-dir "$GW3_RUN/eval_last_$GW3_TAG"
cat "$GW3_RUN/summary.json"
cat "$GW3_RUN/eval_best_$GW3_TAG/report.md"
cat "$GW3_RUN/eval_last_$GW3_TAG/report.md"
```

若best和last模型张量相同，第二条自动复用第一条的新组预测，记录new_inference_calls为0；旧三组始终为0。若模型张量不同，只评价尚未缓存的模型及其干预。不删除已有评估目录，另设时间标签即可。

重点回传summary.json与两个report.md。机制统计查看mechanisms.json，分组看groups.csv，逐时距看horizons.csv。保留所有负收益；五轮结果不等同于超过原模型成熟最佳检查点。

## 已完成五轮后的下一步

若已完成上述训练/评估，直接使用[只读诊断工具](igstgnn_receiver_match_gw3_diagnosis.md)，复用现有制品定位相对GW2的损害。只运行新增的诊断测试与诊断命令，不重复已有检查、训练或推理。
