# GW7 服务器交付

发布分支：`research/graph-interaction-gw7`。基于GW6提交 `42e70c5b20b45ff1f94f1369da6d764614cff77b`，保留GW6及其历史执行源码和原有身份链。

本次新增两部分：服务器上的GW6第18/20轮分支诊断，以及GW7四臂联合训练筛查。模型、Python导入/语法和单元测试没有在本地执行；服务器运行日志才确认可运行性，完成预算不代表精度改进。

## 第一步：只运行分支诊断

先激活原来的 `igstgnn` 环境。从包含 `IGSTGNN-GW6` 和 `data` 的工作目录创建独立checkout：

```bash
git clone --single-branch --branch research/graph-interaction-gw7 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW7
cd IGSTGNN-GW7
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export GW7_GW6_RUN="$(realpath ../IGSTGNN-GW6/experiments/chronological_runs/contra_original_gw6_joint_s2025)"
export GW7_DATA="$(realpath ../data/xtraffic/Contra_Costa)"
export GW7_TAG="$(date +%Y%m%d_%H%M%S)"
export GW7_DIAGNOSTIC_DIR="$GW7_GW6_RUN/layers_$GW7_TAG"
bash experiments/graph_interaction_gw7/run.sh diagnose
```

launcher在没有Slurm allocation时提交完整作业。已获得GPU作业节点且不使用Slurm时，可设置 `GW7_DIRECT=1`；不能在登录节点用它绕过资源分配。`diagnose`完成后退出，不会自动训练GW7。所有环境变量应在当前shell导出，子脚本中的设置不会回传到父shell。

诊断只读已完成20轮GW6的身份、评价归档和JOINT BEST/LAST权重，使用相同验证批次，不进行训练、优化器更新、BN重估或测试集评价。输出目录必须全新；失败后保留现场并换新tag/目录重试。

```bash
cat "$GW7_DIAGNOSTIC_DIR/report.md"
cat "$GW7_DIAGNOSTIC_DIR/summary.json"
```

报告比较每层DIF、INH/TIID、融合隐变量和输出头，包含固定输入下的输出头2×2交叉检查、预测归档回放、实际使用BN的只读buffer变化及认证输入哈希。混合检查点的输出头实验是条件敏感性诊断，不能独立证明某层是根因。

如直接在已有GPU allocation中调用诊断Python入口：

```bash
python -m experiments.graph_memory_gw6.diagnose_layers \
  --run-dir "$GW7_GW6_RUN" --data-dir "$GW7_DATA" \
  --output-dir "$GW7_DIAGNOSTIC_DIR" --device cuda:0 \
  --threads 3 --batches 2 --batch-size 48
```

先审阅此报告；若发现实现或评估状态问题，应先修复并重新核对已有GW6。诊断完成标志只说明检查执行成功，不等于批准某个根因解释。

## 第二步：四臂筛查

保留上述source/data/diagnostic环境变量。在审阅诊断后显式启动：

```bash
export GW7_RUN="$PWD/experiments/chronological_runs/contra_original_gw7_interaction_s2025"
export GW7_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/graph_interaction_gw7/run.sh all
```

`all`执行交付字节核验、服务器单元测试、真实批次无更新前反向预检、四臂各5轮以及BEST/LAST评价。`GW7_DIAGNOSTIC_DIR`必须指向已完成且与当前GW6来源一致的诊断目录；不会重新猜选一次诊断或自动追加训练预算。

| 臂 | 结构 |
|---|---|
| R_CONTINUE | 完整原生递推模型继续训练 |
| R_MLP | 原生模型加历史状态/池化记忆MLP增量 |
| R_MEMORY | 原生模型加历史均值/变化两路普通图记忆增量 |
| R_INTERACTION_MEMORY | 同样两路读取，仅增加图源×图阶交互投影 |

四臂从同一认证R20权重开始，原生DIF forecast保持可训练；共同新增5轮、full12损失、数据顺序和优化器策略。MEMORY与INTERACTION的活跃参数严格相等；MLP实报容量差异。BEST包含第0轮共同起点，另保存固定第5轮LAST。新增阶段统一重置优化器，中断恢复不重置。

每轮记录固定批次的基准预测、增量及总预测变化，避免把可训练主干的漂移误归给新分支。逐轮记录、summary和BEST/LAST评价目录由launcher输出。

中断恢复使用同一来源、诊断目录、种子和模型选项：

```bash
export GW7_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/graph_interaction_gw7/run.sh resume
```

进程被杀后从最后一个完整epoch恢复；未提交的epoch重放。`active.lock`由操作系统锁保护，进程结束时释放，文件存在本身不代表仍在运行。不要删除checkpoint或重置优化器来续跑，也不把本次5轮自动扩展为更多轮。

## 判据与解释

筛查门槛预设为：INTERACTION的BEST pooled MAE相对同次R_CONTINUE与R_MEMORY均改善至少0.5%，且优于MLP；LAST相对自身BEST的MAE退化不超过1%，并实报相对对照LAST的水平；真实值>=314组及7—12步的MAE均不劣于同次R_CONTINUE。这些是工程筛查门槛，不是统计显著性。

评价保留共同有效mask、逐时距、真实值高值组、窗口P95、signed bias和可认证的历史固定困难集。历史GW4/GW6成绩不是本次新增预算对照。真实值分组不进入模型。输出仅为反复使用的开发验证集筛查，不是独立测试结论。

若普通memory或MLP已达到相同效果，不保留交互结构的强贡献主张。通过后另行冻结配置，开展配对多种子及独立确认；launcher不自动运行确认阶段。

结构、性质及失效条件见 [GW7设计说明](docs/igstgnn_graph_interaction_gw7_design.md)。新增源码在 `experiments/graph_interaction_gw7/`；分支诊断在 `experiments/graph_memory_gw6/diagnose_layers.py`。`verify_gw7_bundle.py`只检查文件字节，不能替代服务器测试。
