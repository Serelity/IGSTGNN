# M3.0：事故条件容量到有向潜状态交换

2026-10-10。按[V2.1开发约定](incident_capacity_propagation_v21_contract.md)实现独立空间算子。本轮完成容量—交换—状态演化接口及工程验收，尚未接入IGSTGNN、读取真实目标或进行预测效果比较。

## 已完成与证据

- 共享边交换更新、合流接收预算、分岔及出口发送预算、显式开放入口和零分母处理。
- 有界Bernstein容量轨迹、固定kappa=1、历史/报告条件混合；关闭新增报告保留历史动态容量。
- 同容量入口的普通有向递推；不额外读取报告摘要。算子级参数为受限0、普通244，完整系统参数匹配须在M4完成，当前不构成公平性能实验。
- 明确截止点、内部子步和目标窗口：从t0到+65分钟递推，首个Y窗口[+5,+10)，末个[+60,+65)。输出`[B,12,N,16]`。
- 不合格节点保留轴、直接融合特征为零；直接报告支持之外的上游节点仍可收到传播。

新增21项测试与原26项容量回归共**47项通过，无跳过**，完整GPU Bash流程退出0。另在与服务器相同的**Torch 2.3.1+cu121**版本上运行21项新增测试与GPU检查，均通过。完整47项流程的本地环境为Python3.10.16、Torch2.8.0+cu126；兼容检查为Python3.11.14、Torch2.3.1+cu121。两者GPU均为本机RTX4060 Laptop，不能写成服务器V100结果。兼容环境发出已有NumPy桥接警告，新算子和检查器不依赖该桥接；未修改或安装环境包。

测试覆盖独立单链计算、合流/分岔/边界预算、随机四通道图不变区间及潜收支、瓶颈上游传播与恢复、低需求精确零响应、接收瓶颈屏蔽容量梯度、边/节点重排、四控制系数梯度与有限差分、保存和优化器续跑、目标时间轴、普通参照及CUDA确定性反向。

合成工程记录：

| 检查 | 实测 |
|---|---:|
| 4站场景最大潜收支残差 | 4.47e-8 |
| 496节点合成图最大潜收支残差 | 2.38e-7 |
| 固定参数，四/八子步最大状态差 | 0.008565 |
| 低需求容量变化后的状态差 | 精确0 |
| 保存重放状态差 | 精确0 |
| 恢复后下一次更新参数差 | 精确0 |

子步差的工程门槛为0.05潜单位，仅是固定合成场景的数值敏感性筛查，不是交通精度阈值或完整收敛证明。单单元另与独立解析解比较，4→8→16步误差递减。

本地完整产物为`experiments/chronological_runs/contra_exchange_m30_20261010_105728_L7pbFW/check`，Torch2.3.1兼容产物为`experiments/chronological_runs/contra_exchange_m30_torch231_v1/check`。[机器证据](incident_capacity_exchange_m30_20261010.json)记录两次运行、源码哈希与测试/退出状态。大产物和合成checkpoint不提交Git。

496节点测试采用人工链式连接，明确标记`synthetic_only`；不是原496站道路拓扑，也没有认证全网联合资格。PHYSICAL_CONTRACT_REQUIRED保持不变。

## 服务器运行

先在服务器单独更新`research/chronological-tiid`分支，Git不写入运行脚本。在原环境和已有Slurm分配内执行：

```bash
EXCHANGE_DEVICE=cuda:0 bash experiments/chronological/run_incident_capacity_exchange.sh
```

默认Python是当前环境的`python`。若不依赖激活环境：

```bash
EXCHANGE_PYTHON=/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python \
EXCHANGE_DEVICE=cuda:0 bash experiments/chronological/run_incident_capacity_exchange.sh
```

或从服务器仓库目录提交原V100分区：

```bash
sbatch --partition=gpu_v100 --job-name=capacity_m30 --nodes=1 --ntasks=1 \
  --cpus-per-task=3 --gres=gpu:1 --mem=8G --time=00:10:00 \
  --output=slurm-%x-%j.out \
  --wrap='EXCHANGE_PYTHON=/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python EXCHANGE_DEVICE=cuda:0 bash experiments/chronological/run_incident_capacity_exchange.sh'
```

入口无参数，不需下载候选X或其他数据包，无Git、下载、安装或内嵌Slurm提交。每次创建独立目录，保存日志、退出码、合成响应、checkpoint和`check/summary.json`。成功时显示`M30_ENGINEERING_CHECK_PASS`、`Workflow exit code: 0`。

这一步可复核工程环境，不需要服务器再跑一次才能继续开发。下一步优先补齐真实有向图、来源和可见报告集合资格，再实现M4历史到截止点的状态估计、共享条件容量、边界预测、辅助观测及零初始化融合。保留完整原IGSTGNN事故路径；首轮监督仍采用经资格审查的原3604/917清单。正式预测收益、局部物理桥接与大部分道路联合覆盖另行验收。
