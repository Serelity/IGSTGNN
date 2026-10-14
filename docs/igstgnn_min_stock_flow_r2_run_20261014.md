# MIN-R2：服务器运行与回读

## 🚀 更新与首轮

Git命令与实验脚本分开：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git pull --ff-only
```

复用已有MIN-I A组最佳检查点，只训练两个新候选：

```bash
bash experiments/chronological/run_min_stock_flow.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_stock_flow_r2_bs64_seed2025 \
  --epochs 1
```

在已有Slurm资源中直接运行；不在Slurm资源中时自动sbatch。无`--mem`，脚本内没有Git。继承源batch64与threads3，不修改源数据、源检查点及旧实验。

## 🔄 绝对轮次续跑

先看首轮接通与机制输出。需要继续时，绝对目标轮次改为10：

```bash
bash experiments/chronological/run_min_stock_flow.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_stock_flow_r2_bs64_seed2025 \
  --epochs 10
```

若一组中断，原命令会跳过另一组已完成轮次；仅运行某个已准备worker也可以：

```bash
bash experiments/chronological/run_min_stock_flow.sh \
  --source-run experiments/chronological_runs/contra_min_i_v1_bs64_seed2025 \
  --run-dir experiments/chronological_runs/contra_min_stock_flow_r2_bs64_seed2025 \
  --worker location \
  --epochs 10
```

R2必须用新目录。删除旧算子后只迁移兼容编码权重，新候选初始化不等于原MIN。源MIN不用续训。

## 📖 回读

```bash
cat experiments/chronological_runs/contra_min_stock_flow_r2_bs64_seed2025/comparison.json
cat experiments/chronological_runs/contra_min_stock_flow_r2_bs64_seed2025/report.md
```

| 比较 | 含义 |
|---|---|
| 新候选ON vs 原MIN | 结构替换与新增联合优化的整体效果 |
| 新候选ON vs 同检查点条件OFF | 同一个新解码器对条件服务路径的依赖，不是删模块后重训效果 |
| 同预算location vs state | 同支持M下D数值增量，含额外小投影参数，不是事故有无对照 |

同时看MAE、RMSE、MAPE、低流量、高估/低估组、固定困难窗口，以及`best_validation.mechanisms`的服务主导数、节点/开放账本误差和外部积压。非主导区服务调整不产生通量变化可以是正确行为。

`training_flow_scale.json`保存训练X拟合尺度、峰值覆盖及扩展站点；`identity.json`保存图证据和迁移证书。候选覆盖476站，联合物理认证0站；预测全496站不等于全路网已完成物理标定。

原MIN只在首次准备时回放并缓存，后续worker与续跑复用。没有MIN-only训练、每batch教师重复前向或测试集评价。

## 🧪 工程验收范围

工程入口复用已有4训练/4验证窗口、完整496站源检查点，继承batch2，验证新结构、文件保护、梯度、配对恢复及续跑。单元测试另验证连续2轮与1轮续至2轮的模型、优化器、随机状态和样本顺序一致。

服务器正式实验使用作者3909训练/837验证窗口。工程测试误差不用于评价预测收益。
