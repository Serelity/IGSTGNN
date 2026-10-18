# GW6 服务器源码

本分支实现历史图消息记忆的直接多时距 DIF 解码。源码基于 GW5 提交 `b7386c7931f3dc88b34b8fccab09372be7090749`；历史模型和执行源码的 Git blob 保持不变，新增源码在 `experiments/graph_memory_gw6/`。发布分支为 `research/graph-memory-gw6`。

默认 `joint` 主模型重新训练整个可用主干，以历史 source×order×window 消息直接产生十二时距 DIF 表征，保留历史 backcast、INH 和原 TIID 四块/gap3 输出语义。默认新训练四臂：`RECURRENT_CONTROL`、`POOLED_MEMORY`、`ADDITIVE_MEMORY`、`JOINT_MEMORY`，从同一个认证 GW4 接收组 BEST 检查点开始，各完成相同新增预算。

`probe` 是独立的冻结 GW4 机制探针；零输出初始化能回退原预测，不能代替主模型联合训练。默认 full12 损失在所有臂共同生效；`native` curriculum 需要独立运行身份。

本次已执行的是文件字节与 Git 发布检查。Python 导入/语法、单元测试、真实 batch48 前反向、训练和性能评估均需服务器执行。没有新精度结果；`COMPLETE` 只表示完成预算，不能当成优于基线。

## 在服务器运行

先在原 GW5 目录、保留正确 `$GW5_RUN` 的终端捕获来源：

```bash
export GW6_GW5_RUN="$(realpath "$GW5_RUN")"
test -f "$GW6_GW5_RUN/identity.json"
```

在原 GW5 目录记录好来源后，拉取独立 GW6 checkout：

```bash
cd ..
git clone --single-branch --branch research/graph-memory-gw6 https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW6
cd IGSTGNN-GW6
conda activate igstgnn
export PYTHON_BIN="$(command -v python)"
export GW6_DATA=/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa
export GW6_RUN="$PWD/experiments/chronological_runs/contra_original_gw6_joint_s2025"
export GW6_EPOCHS=5
bash experiments/graph_memory_gw6/run.sh all
```

`all` 在同一 GPU 作业内依次核对交付哈希、运行三个测试模块、真实 batch48 预检、四臂训练、BEST/LAST 评估。没有 Slurm allocation 时提交整个序列，不会在提交返回后抢先评估。已有 GPU allocation 时直接执行；不得在登录节点设置 `GW6_DIRECT=1`。

如已更换终端，`GW6_GW5_RUN` 需设为实际完成的 GW5 实验绝对路径。旧 GW1/GW2/GW4/GW5 检查点、身份与评估缓存需仍可读；缺少证据时停止，不重跑或修改旧实验。

中断恢复：保持原目录与身份参数，更新 tag 后运行：

```bash
export GW6_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/graph_memory_gw6/run.sh resume
```

`GW6_EPOCHS` 是本轮目标总新增轮数。详细参数、可选消融、探针及结果读取见 `docs/igstgnn_graph_memory_gw6_server.md`。

## 主要源码

- `model.py`：逐层历史消息提取、分块读取、直接十二时距预测、joint/probe 和结构消融。
- `reference.py`：GW4/GW5 证据链与共同起点认证。
- `train.py`：相同新预算、MAE 选模、原子 epoch 保存与恢复。
- `verify.py`：真实批量的任务梯度与资源预检，不做优化器更新。
- `evaluate.py`：共享真值轴、原指标回放、历史参考与本轮对照分列、困难集/偏差/逐时距报告。

本分支不包含数据或服务器检查点。已有 GW6 checkout 更新时执行 `git fetch origin` 和 `git merge --ff-only origin/research/graph-memory-gw6`；运行期间保持源码固定。

## 两项逐位回归检查失败后的更新

初版的递推控制/冻结探针测试把 GW4 原有的“仅残差可训练”参考状态，与 GW6 的“全骨干可训练”或“全骨干冻结”状态直接比较。即使在 `no_grad()` 下，部分 PyTorch 版本的高维 Linear/matmul 仍可能根据参数 `requires_grad` 选择不同计算路径，见 [ATen should_fold 源码](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/LinearAlgebra.cpp#L1780-L1833)。更新后的参考模型先对齐各阶段的参数训练状态和 eval 状态，仍使用独立模型副本及严格 `torch.equal`；首层历史消息提取也增加逐位检查。若仍失败，断言会给出 PyTorch 版本、差异元素数、最大绝对误差及首个差异值。具体服务器差异仍需更新后复验，不能仅凭旧断言认定是浮点舍入。

该更新只修改测试和说明，模型、训练策略和历史基准源码保持原样；逐位合同适用于相同执行状态。更新交付哈希后，先在服务器复验：

```bash
git fetch origin
git merge --ff-only origin/research/graph-memory-gw6
python verify_gw6_bundle.py
python -m unittest tests.test_graph_memory_gw6_model tests.test_graph_memory_gw6_workflow tests.test_graph_memory_gw6_evaluation -v
```

本地未执行这些 Python 模型检查。服务器测试通过后，沿用既有环境变量，更新 `GW6_TAG` 再运行 `bash experiments/graph_memory_gw6/run.sh all`。初版 `all` 在单元测试失败时即停止，此时尚未开始真实批量预检或训练。
