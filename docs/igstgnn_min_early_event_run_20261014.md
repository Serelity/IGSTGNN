# MIN-E1服务器运行与结果回读

_2026-10-14。原作者发布数据；默认新训练STATE与EVENT，原MIN不重训。_

---

## 📥 更新服务器代码

以下git命令单独运行，不在任何训练Bash脚本内：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

若本地改动导致pull拒绝，保留错误输出，不执行reset或清理既有实验目录。

## 🧪 先作一次工程检查

```bash
bash experiments/chronological/run_min_early_event.sh \
  --run-dir experiments/chronological_runs/contra_min_early_e1_check_seed2025 \
  --check \
  --evaluate-after
```

只用4训练/4验证样本、batch2，各新组1轮，并运行最佳回放与配对评估。该检查不构成预测效果证据；正式运行必须使用独立目录。已经通过的同版本检查不必重复。

## 🚀 正式先跑5轮并自动评估

```bash
bash experiments/chronological/run_min_early_event.sh \
  --run-dir experiments/chronological_runs/contra_min_early_e1_bs64_seed2025 \
  --epochs 5 \
  --evaluate-after
```

从头联合训练STATE、EVENT各5轮，相同初始主干、同注册参数数量、同训练顺序。每组441,324参数，新增5,468。无已有MIN训练worker，也不从已有MIN最佳权重热启动。

脚本在登录节点自动sbatch；已在Slurm分配内则直接执行。未申请`--mem`，CPU数3，GPU数1。仓库工作目录与日志均使用绝对路径。训练结束后同一作业内先回放最佳权重，再运行事故内容与forecast审计；不会先异步提交另一个评估job。

阶段轮数是绝对目标。训练中断后运行同一命令即可续跑；已完成的组不重复训练。已训练5轮后改为`--epochs 10`表示续到总10轮。已有目录不得更改batch、seed、组集合、协议或模型源码后强行续跑。

## 🔎 单独评估已有最佳权重

已经完成训练、只想分析时运行：

```bash
bash experiments/chronological/run_min_early_event_evaluation.sh \
  --run-dir experiments/chronological_runs/contra_min_early_e1_bs64_seed2025
```

该入口不训练、不评价测试集。默认5种正常/关闭/内容状态，加12种forecast直接读出干预。需要只做事故内容初筛时可加`--skip-forecast-audit`；已有完整评估制品优先回读，不为相同问题重复推理。

## 📄 回读报告

等控制台出现`MIN_EARLY_EVENT_EVALUATION_PASS`后，使用其打印的评估目录，或在源目录中读取最近一次成功报告：

```bash
RUN_DIR=experiments/chronological_runs/contra_min_early_e1_bs64_seed2025
EVAL_DIR="$(find "$RUN_DIR" -maxdepth 2 -name summary.json -path '*/early_event_evaluation_*/*' -printf '%h\n' | sort | tail -n 1)"
test -n "$EVAL_DIR"
cat "$EVAL_DIR/summary.json"
cat "$EVAL_DIR/report.md"
cat "$EVAL_DIR/mechanisms.json"
cat "$EVAL_DIR/comparability.json"
```

变量赋值与cat分别成行，避免shell在赋值生效前将路径展开为空。若某次评估失败，查看其`failure.json`与作业日志，不把不完整目录的中间产物当完成结果。

## 📐 结果解释规则

首轮主要比较相同预算下的EVENT与STATE，读取完整曲线而非仅末轮。已有原MIN最佳28轮只是历史参照，不能将5轮新模型与它直接作同预算结论。

EVENT_OFF保留EVENT训练后的主干；它不等于旧MIN。位置重排移动完整D及其非零支持，可能离开自然道路配置；语义UNK保留M/D。forecast移除仅测试最终直接读出，不证明整个分支或整层可删。

模型结构与公式见[模型方案](igstgnn_min_early_event_plan_20261014.md)，指标、保护与后续决策见[评估方案](igstgnn_min_early_event_evaluation_20261014.md)。
