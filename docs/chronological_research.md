# 日期划分研究开发入口

分支：`research/chronological-tiid`。状态：`conditional_development`。

这里接续已完成的IGSTGNN复现，验证新的数据加载、事件输入和A/B/C时间权重能否在真实batch上正确运行。**A是采用共同新输入约定的研究基线，不等同论文原表设置；这里没有全量训练后的精度结论。** 原 `experiments/IGSTGNN/main.py` 的默认行为和论文对齐版checkpoint格式保留。

## 固定数据与信息范围

- 数据源：`gpxlcj/xtraffic` v8、2023、Contra496站，按照station_id回连原始轴。
- 清单：1—8月训练3604条；9—10月验证917条；11—12月测试894条暂仅保留清单，本入口不构建或加载test目标。
- `r`为首次报告名义时间；`T=floor_5min(r)+5min`；X标签T−65…T−10的12槽，Y标签T+5…T+60的12槽，间隔5分钟。26槽原始包的`0:12`为X、`14:26`为Y，中间2槽不用。
- 训练统计只来自去重的train X原始有限非负流量；输入缺失用每站训练均值填补，目标无效点排除。真实0有效，不使用旧stats或“0=缺失”的评价代码。
- 小批训练loss采用原始尺度pooled MAE；评价分别输出各预测步、逐步宏平均和pooled指标。主评价字段是`mae_macro`；MAPE仅在目标>0上补充并记录计数。

原始作者确认旧2023排序修复，v8是当前修订源；其每值真实时间、时区、DST与数据延迟尚未独立认证。当前用名义日历和“交通聚合完成后至多5分钟可得”的显式假设。前序旧验证与部分全年交通曾被查看，test不能声称完全盲测。

`report_location_v1`进一步**假定首报已知当前记录的Fwy/Direction/Abs PM**；源表没有首报版本快照，因此仍为有条件开发。事件向量只用`T−r`报告后年龄和T的日内/星期共享嵌入，不用Type、Description、Holiday、最终duration或旧position索引。静态传感器属性附加编码在A/B/C中共同关闭。

空间向量为`D=[0,w,b]`：同数字Fwy和同Direction、`abs(sensorPM-eventPM)<=10`时，`w=(exp(-delta²/2)-exp(-50))/(1-exp(-50))`，`b=(eventPM>sensorPM)`；否则0。第1列有意置0，第3列仅为里程大小关系，不宣称真实上下游或米制距离。ICSF及TIID上下文沿用论文对齐机制。

## A/B/C只改变时间系数

令`g(h)=exp(-h²/2)`，h=1…12对应T+5…T+60的输出槽。

| 变体 | 系数 | 相对A新增参数 |
|---|---|---:|
| fixed / A | g(h) | 0 |
| shared / B | g(h)+tanh(b[h]) | 12 |
| conditioned / C | g(h)+tanh(b[h]+MLP(z)[h]) | 268 |

每节点`z=[历史标准化流量均值,最后值,后3步均值−前3步均值]`。MLP为3→16→12、ReLU，末层无bias且weight初始化0；b初始化0。由共同b承担逐步偏移，避免C多一套显式截距。系数可为负或非单调，属于预测残差权重，不解释成已识别的物理衰减。

检查脚本逐tensor复制并核验A/B/C共同初值；eval输出先比对，再在同一个按时间最早的真实train批次上各更新两次，检查loss、梯度、参数及时间分支自身的H12梯度。C第二步还检查早层loss梯度，防止把Adam weight decay造成的变化误当作学习。真实val仅取固定前两条检查输出和评价代码，不排名、不当作完整验证结果。

## 服务器：数据包及Python检查命令

本地实测已通过：496站完整train/val包、59项回归测试，以及CPU上A/B/C同批次各两次反向更新。参数量为443645／443657／443913；共同state逐tensor一致，初始预测差均为0。三组更新总计约5.03秒（不含校验包、构模和验证前向），不是完整训练耗时。B/C的H12损失梯度非零，C第二步早层损失梯度非零；这些均在优化器执行前检查。

本轮准备的独立文件为`Contra_Costa_v8_trainval_dev_20260911.tar.gz`，89,340,908字节（约89.34MB），包含原始窗口、训练统计、事件上下文、固定邻接、清单和来源指纹，不包含下载缓存。25个文件及其中23个载荷指纹已从压缩包验证。数据和结果不随Git提交。

压缩包SHA256：`67d21de28a9eb86c69e6cb249961eaf8e347abb95d9d25d3820c0e67ad9452ac`。

把此文件上传到服务器`/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/`。代码更新命令：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git fetch origin
git switch research/chronological-tiid
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
```

首次解包，目标使用独立新目录，保留旧数据：

```bash
python -m tarfile -e ../data/Contra_Costa_v8_trainval_dev_20260911.tar.gz ../data/chronological/Contra_Costa_v8_dev
```

在平台已分配的计算任务中运行，CPU即可完成本轮检查：

```bash
python experiments/chronological/smoke.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_cpu_smoke_01 \
  --device cpu \
  --bs 2 \
  --steps 2 \
  --seed 2025
```

若已分配GPU任务，可把`--device cpu`改成`--device cuda:0`并使用新的output-dir。本轮无需为几步检查申请长时GPU；bs=2不能证明正式bs=48的显存或吞吐。已有服务器Python3.10/PyTorch2.3.1+cu121/NumPy1.24.4可先运行；本地实测版本见运行JSON，不为这一步重装原环境。

成功输出目录含`summary.json`与`batch_predictions.npz`，状态为`CONDITIONAL_REAL_BATCH_SMOKE_PASS`。输出目录已存在就报错，重跑使用新目录名。该脚本只做工程更新，**没有保存可用于正式评价的训练模型，也没有运行一整轮或完整验证**。

## 本地重建与回归

`build_data.py`不依赖旧研究脚本代码，但需要已有的冻结清单、协议、原7站小包、传感器CSV及v8 node_order文件。它按当前本地项目目录结构定位这些输入；单独git clone并不包含这些数据证据。服务器直接使用上面的准备包，无需重复下载。

原有训练依赖不变；只有源数据下载器额外需要requests，可用`python -m pip install -r requirements-research.txt`安装。已有环境直接运行准备包的smoke不需要额外网络依赖。

在完整本地工作目录中，首次构建使用：

```bash
python -m experiments.chronological.build_data --preflight
python -m experiments.chronological.build_data --workers 12
```

请求均验证HTTP206、精确字节范围、长度和SHA256。失败请求不当作源缺失；行缓存可复用，输入指纹不同或已有完成产物时拒绝覆盖。源码中的默认输出路径为独立研究目录，不覆盖发布`incident_all.npy`。

回归检查：

```bash
python -m unittest discover -s tests -v
```

正式训练仍需完成数据时间/事件字段可信性决策、最近邻方法核查及训练口径冻结。A/B/C主比较要在共同协议下从头训练，条件化通过后再做匹配容量D控制和重复实验。

来源：[数据作者错序修复说明](https://github.com/XAITraffic/XTraffic/issues/13#issuecomment-4932757405)、[报告分箱答复](https://github.com/XAITraffic/XTraffic/issues/11#issuecomment-2971180398)。实验设计过程采用experimental-design技能的共同对照和重复单位原则；该流程不是交通模型效果证据。
