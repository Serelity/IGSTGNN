# M4.3续跑身份诊断修正

后续修正：用户无法选择GPU型号，已增加[仅GPU型号变化的显式兼容续跑](incident_capacity_gpu_migration_20261010.md)。加`--allow-gpu-name-change`可记录当前已确认的唯一差异并继续；默认严格检查仍保留。以下匹配原型号的方法只适用于能够选择节点的环境。

2026-10-10。用户从gpu14转到gpu05，继续同一运行至绝对epoch10时，原训练器在`Run identity changed`检查处停止。没有第6轮训练记录；前面的六组表格是续跑前的第5轮快照，不能作为10轮结果。原检查发生在构建共同初始化之后、创建任何组的优化更新之前，已有5轮检查点应继续保留。

## 已确认与待确认

已确认原续跑器的源、协议、输入文件和六组检查点回读已通过。完整训练身份还包含Python/Torch/NumPy/CUDA版本、GPU型号、确定性设置以及共同主干和分支初始化哈希。原报错没有打印不同字段，不能仅凭换了主机就断定原因。

后续用户回传`continuation_20261010_172528_9a5d7e`诊断输出，逐字段差异只有一项：`environment.gpu`由原来的`Tesla V100-SXM2-32GB`变为`Tesla V100-PCIE-32GB`。这确认当前拒绝由GPU型号字段不符触发；没有报告其他身份差异，仍在任何组的训练更新前退出，尚无10轮结果。证据来自用户粘贴的控制台输出，未独立读取服务器JSON。上方首次报错本身未打印差异，机器记录分别保留两次证据。

主机名本身不在原身份中；同型号GPU可以位于不同节点。命令显式使用igstgnn环境的Python，所以shell提示符`(base)`本身不能证明用了错误解释器。

## 实现

新增`experiments/chronological/resume_incident_capacity_checked.py`，由原续跑入口作为子进程调用。它执行未改动的原训练器，在原身份比较处记录逐字段的旧值和新值；身份不一致时原拒绝条件继续生效。

正常模式：身份相符才继续原训练过程；不符则保存`identity_diagnostic.json`并打印`RESUME_IDENTITY_MISMATCH`与差异路径。原身份、各组检查点及共同初始化哈希不会被改写，也不放宽比较条件。新入口和诊断器哈希分别保存，不进入旧训练源指纹。

新增只读模式`--identity-audit-only`，重建当前完整训练身份后立即退出。除诊断记录外不执行训练更新或原训练产物写入；即使身份相符也停止。既有`--audit-only`仍只检查已保存产物，不重建GPU环境与初始化。两种模式互斥。

每次调用仍使用独立`continuation_.../`目录与排他锁，避免覆盖历史诊断。诊断保存旧/新完整身份与字段差异；初始化哈希差异也会列出，不将其误报为单纯GPU名称变化。旧训练器、模型、协议及17个SOURCE_FILES文件保持原字节。

## 服务器操作

独立更新代码：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only origin research/chronological-tiid
```

在已分配GPU的终端重试原10轮命令：

```bash
/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python -u \
  experiments/chronological/continue_incident_capacity.py \
  --run-dir experiments/chronological_runs/contra_training_m42_20261010_153544 \
  --history-dir experiments/chronological_runs/contra_v11a_multichannel_history_02 \
  --epochs 10
```

若只想检查差异，在命令后加`--identity-audit-only`。正常模式相符则续训，不符则在任何更新前列出差异。

若仅GPU型号不同，先申请与原身份相同型号的GPU，再用同一命令续跑，不要求固定gpu14节点。若版本不同，恢复原环境；若初始化哈希不同，需要进一步核对其来源，不能通过编辑identity或检查点消除报错。跨硬件迁移应另设验证与记录，当前不作为同身份续跑自动放行。

当前已确认仅GPU型号不同。可回到登录节点，优先申请原来实际运行过的gpu14；节点是否可分配及当前设备以调度器和`nvidia-smi`输出为准。按项目已有分区及资源设置申请交互终端：

```bash
srun --partition=gpu_v100 --nodelist=gpu14 \
  --nodes=1 --ntasks=1 --cpus-per-task=3 --gres=gpu:1 \
  --time=02:00:00 --pty bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

确认分配设备为`Tesla V100-SXM2-32GB`后，进入仓库运行上方Python续跑命令。若节点忙则等待分配，或使用管理员提供的其他SXM2节点；不猜测集群未核实的GPU类型/constraint名称。现有运行代码无需为此次字段差异再修改。

本次修正没有新增实验组，也没有开始淘汰对照；六组仍应先完成共同10轮预算，再比较后续开发优先级。

## 验收

新增7项身份诊断测试和既有66项回归全部通过，共73项、无跳过。覆盖GPU、初始化哈希及缺失字段的差异定位、正常严格拒绝、相符身份续行、只读模式停止、意外训练路径阻断、原产物和函数恢复、诊断文件防覆盖及参数传递。

另使用真实原数据首4条的隔离检查点副本验证：只读诊断前后31个原产物哈希一致；匹配环境下通过新入口将六组从epoch2续至epoch3，每组累计6次更新。人工GPU名称差异只注入独立测试夹具，确认在创建组产物前拒绝。该受控夹具不证明服务器差异就是GPU名称。本地子集结果仅作为工程证据，不支持预测增益结论。17个冻结训练源文件及协议哈希与原检查点身份相符。验收机器记录另存`incident_capacity_resume_identity_fix_20261010.json`。
