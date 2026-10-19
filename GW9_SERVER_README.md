# GW9 服务器交付：保留 DIF/INH 身份的受控读出

发布分支：`research/branch-readout-gw9`。继承修复历史回放后的 GW8 提交 `12e5f67fd5d7e513f7e8bf03956e9210da8b4bdf`，只新增 GW9 文件，保留认证历史源码的原始字节。

本版已实现，尚未在服务器执行新增测试、真实 batch 48 预检和研究实验。本地遵守不运行 Python、导入、语法检查、模型或测试的约定，只完成文本、Git、文件字节核对和人工静态审阅。以下精确恢复、梯度与筛查要求均需实际服务器日志确认。

## 问题与结构

原生五层 DIF 和 INH 预测隐状态分别按原顺序求和，得到 D、I，再相加并经过 TIID、共享非线性输出头。GW9 检验：在完整保留当前 R 预测的条件下，差值 Q=D−I 是否仍有总和之外的预测价值。

S 取本次真实前向执行的 TIID 输出，位置在原输出头 ReLU 之前；未调用 TIID 时取 D+I。Q 始终为原始 D−I。代码只安装临时观察 hook，不重新构造图、重新计算事故上下文、重复 TIID 或 dropout，也不改变原预测递推。

| 臂 | 新读出输入 | 比较目的 |
|---|---|---|
| R_CONTINUE | 无新读出 | 保留完整 R |
| SUM_CONTROL | [S; tanh(S)] | 控制额外非线性读出容量 |
| BRANCH_READOUT | [S; tanh(Q)] | 检验分支差异的增量价值 |

两新增头均为 `Linear(512,16,bias=True) → Tanh → Linear(16,3,bias=False)`，各 **8,256 参数**，初始张量相同，最后投影为零。输出按原四个预测块、每块三个通道的 gap=3 顺序映射到完整 12 步，再加到原 R 预测。头权重退化为零应精确恢复 R。

## 两阶段预算

1. **默认只运行 frozen probe。** 冻结并 eval 整个 GW8 R_CONTINUE BEST，两个小头共享每批单次真实主干前向产生的 base/S/Q；不保存特征缓存，不重训主干。每头固定 **3 轮、246 次更新**，BEST 包含阶段零，LAST 为第 3 轮。
2. 仅当 probe 的 `screening.json` 中 `eligible_for_joint_screen` 为 true，才显式执行 joint。三个臂重新从同一 GW8 R BEST 开始，两新头重新零初始化，**不加载 probe 头权重**；全部原参数共同训练，各 **5 轮、410 次更新**，BEST 包含阶段零。

这里 82 batch/轮来自既有认证数据与 batch 48；程序同时检查实际数据身份和更新数。probe 中冻结 R 为零更新参考，不能称为同训练预算的 R 对照；只有第二阶段的 R_CONTINUE 才控制继续训练收益。启动阶段重置 Adam，恢复时恢复完整优化器、调度器、数据顺序和 RNG，总预算不增加。

probe 的 BRANCH BEST MAE 必须相对 SUM 与冻结 R 均至少改善 0.5%，LAST 相对自身 BEST 退化不超过 1%，高值≥314 与 7—12 步组 MAE 不劣于冻结 R；全部通过才设置 `eligible_for_joint_screen=true`。

来源固定为既有 GW8 R_CONTINUE BEST，开发验证 MAE **13.508353274**；GW8 运行身份 SHA-256：

`61aa8b9eab14a6d17315dfdd16db0ea290929dfc883c058b59f01552addc7f09`

BEST 的轮次、文件名和权重 SHA 从认证运行中读取，不根据结果表猜测。认证同时核对完成的 GW8 四臂、修复后的 BEST/LAST 归档、历史回放记录及 GW7/GW6 来源链。请保留历史目录与文件，不编辑 identity、缓存或 checkpoint。

## 克隆并运行第一阶段

在服务器执行：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
git clone --single-branch --branch research/branch-readout-gw9 \
  https://github.com/Serelity/IGSTGNN.git IGSTGNN-GW9
cd IGSTGNN-GW9
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW9_GW8_RUN="$(realpath ../IGSTGNN-GW8/experiments/chronological_runs/contra_original_gw8_screen_s2025)"
export GW9_DATA="$(realpath ../data/xtraffic/Contra_Costa)"
export GW9_PROBE_RUN="$PWD/experiments/chronological_runs/contra_original_gw9_probe_s2025"
export GW9_JOINT_RUN="$PWD/experiments/chronological_runs/contra_original_gw9_joint_s2025"
export GW9_TAG="$(date +%Y%m%d_%H%M%S)"

bash experiments/branch_readout_gw9/run.sh probe
```

没有 Slurm allocation 时，launcher 默认通过 sbatch 申请 GPU。如果已分配 GPU 资源、直接在 gpu05 上运行，可在启动命令前设置：

```bash
export GW9_DIRECT=1
```

`GW9_DIRECT=1` 只表示已获得 GPU 后直接执行，不会申请资源。父 shell 必须 export 这些变量；launcher 内部设置的变量不会返回父 shell。

执行顺序是源码包校验 → 三组新增服务器测试 → 真实 batch 48、零优化器更新的前反向预检 → 两个头固定 3 轮 → 自动生成 probe 报告与锁定筛查。任一步失败即停止，不自动缩小 batch、修改卡尺、追加轮数或启动 joint。

来源 BEST/LAST 自动选择认证匹配的完整归档；需明确指定时，在启动前设置：

```bash
export GW9_GW8_BEST_EVAL="/绝对路径/原GW8运行目录/eval_best_原tag"
export GW9_GW8_LAST_EVAL="/绝对路径/原GW8运行目录/eval_last_原tag"
```

## 查看 probe 结果并决定第二阶段

```bash
cat "$GW9_PROBE_RUN/summary.json"
cat "$GW9_PROBE_RUN/report.md"
cat "$GW9_PROBE_RUN/screening.json"
```

仅当 `eligible_for_joint_screen` 为 true，显式执行：

```bash
export GW9_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/branch_readout_gw9/run.sh joint
```

joint 入口重新认证 probe 与来源。冻结 probe 未通过时会拒绝第二阶段，不能手工改 screening.json 绕过。joint 完成后自动生成 BEST/LAST 评价和最终开发筛查；probe 通过仅说明值得做联合训练，不能直接认定模型改进成功。

joint 开始后保留对应 probe 产物不变。再次运行 probe-evaluate 会发布新的评价证书，已有 joint 绑定的旧证书将无法通过认证；此时不要用重跑 probe 评价处理 joint 的中断，直接使用 joint-resume。

```bash
cat "$GW9_JOINT_RUN/summary.json"
cat "$GW9_JOINT_RUN/eval_best_$GW9_TAG/report.md"
cat "$GW9_JOINT_RUN/eval_last_$GW9_TAG/report.md"
cat "$GW9_JOINT_RUN/screen_$GW9_TAG/screening.json"
```

若新 shell 中变量丢失，先重新执行克隆命令之后的路径 export；忘记 joint tag 可定位现有文件：

```bash
find "$GW9_JOINT_RUN" -maxdepth 2 \( -name report.md -o -name screening.json \) -print
```

## 中断恢复与只重跑评价

保留相同目录、数据、种子和来源。刷新 tag 后按阶段执行：

```bash
export GW9_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/branch_readout_gw9/run.sh probe-resume
```

或：

```bash
export GW9_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/branch_readout_gw9/run.sh joint-resume
```

从最后一个原子提交的完整 epoch 恢复；未完成 epoch 重放，已完成阶段不增加轮数。不要删除 checkpoint、优化器或历史源文件。恢复模式重新执行预检，临时模型与临时训练模式缓冲均被丢弃。

仅重跑已完成 probe 的评价和筛查，不更新任何模型：

```bash
bash experiments/branch_readout_gw9/run.sh probe-evaluate
```

仅重跑已完成 joint 的 BEST/LAST 和筛查，不训练：

```bash
export GW9_TAG="$(date +%Y%m%d_%H%M%S)"
bash experiments/branch_readout_gw9/run.sh evaluate
```

单独预检：`bash experiments/branch_readout_gw9/run.sh check`。之后再调用 probe/joint 会重新预检；默认直接 probe 即可。每次预检报告写入 `${GW9_PROBE_RUN}_verification_${GW9_TAG}_${MODE}.json`；重复同一 mode 前更新 tag，避免覆盖。

## 解释边界

最终 joint 要求 BRANCH_READOUT 的 BEST MAE 相对本次 R 与 SUM 均至少改善 0.5%，LAST 相对自身 BEST 退化不超过 1%，真实值≥314 与 7—12 步组 MAE 不劣于本次 R。失败不追加预算。历史 ORIGINAL/GW2/GW4 表格只复用不可变缓存，不是本次新预算对照。

分支差值不是物理状态、波动强度或因果解释；本版没有利用前期被否定的系统性趋势欠响应假设。分解和融合已有相关研究，本版的研究价值需要受控结果证明，不能提前保证强创新或精度收益。全部阶段均为反复使用开发验证集的单种子筛查，未调用测试集。详细公式、替代解释和失败判据见 [设计说明](docs/igstgnn_branch_readout_gw9_design.md)。

数据装载沿用作者的整体数组文件，会反序列化其中的完整数据对象；GW9 仅迭代训练和开发验证 loader，不迭代测试 loader、不计算测试指标，也不使用测试标签选模型。
