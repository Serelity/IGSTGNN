# GW11 LAST：源模型与新增分支隔离诊断

本次使用已经完成的 GW11 LAST 检查点，分别评价完整模型和关闭新增分支后的完整 `model.source`。固定四个实验臂、八次完整验证，不新增训练、优化器或梯度更新。仅使用原验证集，不评价测试集，不重新开启 GW11 筛选。

本地仅做源码静态检查与字节检查，没有执行 Python、导入、语法检查、测试或模型。新增 47 项服务器测试和模型重放是否通过，以服务器实际输出为准。

## 服务器运行

在保留 GW9、GW10 和 GW11 结果的原 GW9 工作目录更新同一分支。使用原 Conda 环境和与 GW11 相同的 GPU 运行环境；目录中的原文件和历史结果应保留。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9
git pull --ff-only origin research/state-gate-gw11
conda activate igstgnn

export PYTHON_BIN="$(command -v python)"
export GW11_RUN="/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW9/experiments/chronological_runs/contra_gw11_state_gate_s2025_20261019_164846"
export GW11_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW11_ABLATION_DIR="${GW11_RUN}_last_ablation_$(date +%Y%m%d_%H%M%S)"
export GW11_DEVICE="cuda:0"
export GW11_THREADS="3"

bash experiments/state_gate_gw11/run_ablation.sh
```

脚本依次执行源码字节检查、三个新增测试模块，再启动 LAST 诊断。`GW11_RUN` 必须显式设置为含 `identity.json` 的原绝对路径。输入的 GW9、GW10 缓存及掩码审计目录从原身份文件恢复，并核对已绑定的来源；不会通过发现脚本选择最新运行。

本入口不读取 `GW11_RESUME`、`GW11_EVALUATE_ONLY` 或 `GW11_STAGE`；原 shell 中的这些变量无须修改。默认输出目录为 `${GW11_RUN}_last_ablation_时间戳`，也可以显式设置 `GW11_ABLATION_DIR`，但必须使用新的绝对路径。

完成后回传这两份输出：

```bash
cat "$GW11_ABLATION_DIR/report.md"
cat "$GW11_ABLATION_DIR/decomposition.csv"
```

机器可读结果保存在 `analysis.json`；按时距、节点及真实流量区间划分的结果分别保存在：

```bash
cat "$GW11_ABLATION_DIR/by_horizon.csv"
head -n 20 "$GW11_ABLATION_DIR/by_node.csv"
cat "$GW11_ABLATION_DIR/by_flow.csv"
```

## 核对什么

四臂为 `SUM_CONTINUE`、`STATIC_GATE`、`STATE_READOUT`、`STATE_GATE`。每臂先运行完整 LAST（ON），要求预测逐值精确重放原 LAST 缓存；再直接运行同一实例中的完整 `model.source`（OFF）。ON 重放失败或 SUM 的 ON/OFF 预测存在差异时，诊断停止，不放宽容差。

OFF 保留已经联合训练的源模型，包括原骨干和原 SUM 头。关闭的是 GW11 新增分支；程序不清零参数、不改检查点、不冻结参数的 `requires_grad` 标志。八次验证均使用 `eval()` 和无梯度执行，并保持原完整模型验证的执行条件。

既有加载器会整体反序列化原始 object 数据文件，其中包含测试对象；本次仅创建和遍历 `val_loader`，不创建训练或测试 loader。

结果比较包含三部分：原始 `FROZEN_SUM`、LAST 的源模型 OFF、完整 LAST 的 ON。

- OFF 相对原始 FROZEN_SUM 的差异，描述联合训练后的源模型变化。
- ON 相对 OFF 的差异，描述该检查点下新增分支对预测的直接改变。
- ON 相对原始 FROZEN_SUM 的差异，描述完整 LAST 的总变化。

新增分支与源模型经过共同训练，存在共同适应。OFF 比较是检查点上的干预诊断，不能独立证明某个训练机制导致退化，也不能证明冻结源模型后的训练一定有效。全局及分组 MAE、偏差和预测变化共同用于判断下一步实验；平均偏差变负本身不足以证明所有预测只是整体平移。

`decomposition.csv` 报告上述成对变化；`by_horizon.csv`、`by_node.csv`、`by_flow.csv` 用于定位差异集中在哪些时距、节点和流量区间。`analysis.json` 保存全局指标、诊断解释及表格引用；`paired_predictions.npz` 保存配对预测，`metadata.json` 保存来源与运行记录。流量分组使用验证真值，只用于事后评价。

## 失败或中断

本次仅写入新诊断目录，不更新原 `summary.json`、`screening.json` 或任何训练检查点。不存在训练续跑语义。失败或被杀后保留原输出目录，设置一个新的 `GW11_ABLATION_DIR`，重新执行同一个脚本即可；八次验证会从头重放。

若提示来源哈希、运行身份、LAST 缓存或数据身份不匹配，保留目录并回传完整报错。不要修改旧身份文件、旧缓存或放宽精确重放条件。
