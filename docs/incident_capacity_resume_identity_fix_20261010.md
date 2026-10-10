# M4.3续跑身份诊断修正

2026-10-10。用户从gpu14转到gpu05，继续同一运行至绝对epoch10时，原训练器在`Run identity changed`检查处停止。没有第6轮训练记录；前面的六组表格是续跑前的第5轮快照，不能作为10轮结果。原检查发生在构建共同初始化之后、创建任何组的优化更新之前，已有5轮检查点应继续保留。

## 已确认与待确认

已确认原续跑器的源、协议、输入文件和六组检查点回读已通过。完整训练身份还包含Python/Torch/NumPy/CUDA版本、GPU型号、确定性设置以及共同主干和分支初始化哈希。原报错没有打印不同字段，不能仅凭换了主机就断定原因。

当前最需要核对GPU型号与运行环境。主机名本身不在原身份中；同型号GPU可以位于不同节点。命令显式使用igstgnn环境的Python，所以shell提示符`(base)`本身不能证明用了错误解释器。服务器的确切差异须由新诊断回传，当前尚未独立取得。

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

本次修正没有新增实验组，也没有开始淘汰对照；六组仍应先完成共同10轮预算，再比较后续开发优先级。

## 验收

新增7项身份诊断测试和既有66项回归全部通过，共73项、无跳过。覆盖GPU、初始化哈希及缺失字段的差异定位、正常严格拒绝、相符身份续行、只读模式停止、意外训练路径阻断、原产物和函数恢复、诊断文件防覆盖及参数传递。

另使用真实原数据首4条的隔离检查点副本验证：只读诊断前后31个原产物哈希一致；匹配环境下通过新入口将六组从epoch2续至epoch3，每组累计6次更新。人工GPU名称差异只注入独立测试夹具，确认在创建组产物前拒绝。该受控夹具不证明服务器差异就是GPU名称。本地子集结果仅作为工程证据，不支持预测增益结论。17个冻结训练源文件及协议哈希与原检查点身份相符。验收机器记录另存`incident_capacity_resume_identity_fix_20261010.json`。
