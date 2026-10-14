# MIN-R1：服务器运行与已有实验复用

执行计划：[五问题开发计划](igstgnn_min_response_r1_plan_20261014.md)。运行前核对：[已完成实验与复用清单](igstgnn_min_experiment_reuse_ledger_20261014.md)。

**本轮不再训练 MIN-only。**只读加载已有 `contra_min_i_v1_bs64_seed2025/A/last.pt` 的最佳 MIN 权重，在独立目录训练 state/location 两个新候选。候选内部的主干副本参与联合微调，源 MIN 权重及旧预测保持不变。

## 更新代码

Git 与实验 Bash 分开。进入正确仓库后更新本研究分支：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

## 首轮只运行两个新候选

```bash
bash experiments/chronological/run_min_response_joint.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_response_r1_joint_bs64_seed2025 \
  --epochs 1
```

已处于 Slurm 作业内时直接运行；未处于作业内时 Bash 自动用 `sbatch` 提交，并设置正确工作目录。脚本没有 Git 操作，没有 `--mem`；使用源批大小 64 与源线程数。两候选顺序执行，不把两套训练计算图同时放入显存。已有源 MIN 仅用于推理和参照损失，不进入优化器。

首轮通过后可在同一目录续至绝对第十轮：

```bash
bash experiments/chronological/run_min_response_joint.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_response_r1_joint_bs64_seed2025 \
  --epochs 10
```

`--epochs 10` 是每组完成到第十轮，不是在已有轮数上再加十轮。已达到目标且制品完整的组会跳过。中断后再次运行同一条命令，从该组最后成功提交的 epoch 边界恢复；不重跑已完成的整轮。不要用旧 service/frozen 目录承接这个不同结构。

如只需恢复已准备目录中的一个候选，可使用：

```bash
bash experiments/chronological/run_min_response_joint.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_response_r1_joint_bs64_seed2025 \
  --worker location \
  --epochs 10
```

不要在同一目录同时启动两次父流程。入口使用锁防止重复写入；GPU 型号变化只记录环境审计，数据、代码、批大小和源检查点身份仍须一致。

## 回传结果

```bash
cat experiments/chronological_runs/contra_min_response_r1_joint_bs64_seed2025/comparison.json
cat experiments/chronological_runs/contra_min_response_r1_joint_bs64_seed2025/report.md
```

重点看三个比较：候选相对原 MIN 的完整方案变化、location 相对同预算 state 的关系增量、同一个联合检查点 ON 相对 current OFF 的路径作用。后一个 OFF 已经允许主干更新，不能当作旧 MIN，也不能当作现实中的无事故世界。

同时看 MAE/RMSE/MAPE、固定困难窗口、真实流量分层与高估/低估风险。完整记录在 `risk_diagnostics.json`、各组 `history.json` 和 `summary.json`。`source_MIN_training_updates` 必须为 0；候选 `backbone_state_changed` 则应反映主干副本是否更新。

当前只支持源 MIN-I 的 A 最佳检查点，不能仅改路径读取任意其他模型。若未来已有同协议的可复用预测，优先读取；必要检查点回放属于推理，不重新训练。

## 可选冻结诊断与本地工程范围

`run_min_response.sh` 是同一新结构的冻结诊断入口，不是最终优化入口。服务器首轮已直接运行联合候选时，不要求再运行一套完整冻结试验；旧冻结适配器和 service 对照也不重复安排。

`--check` 仅用于已有工程源目录，不能把正式验证集临时截断成工程集。检查模式可在本地 CPU 的四样本工程源上验证启动、保护与续跑，不用于宣称预测改进。正式运行不加 `--check`，不评价测试集。
