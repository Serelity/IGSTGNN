# v12e：节点门控的区域损失对照

## 依据与问题

v12d 在 gpu20 完成，用户于 2026-09-29 回传日志：完整拟合约 18 分 12 秒，退出码 0。
关闭 ICSF 注入使完整事故拟合/选择期全局 MAE 恶化约 0.03329/0.03236；说明不能因为
v12c 改善弱就删除原事故路径。g=0.8 同时改善两个时段的候选早期/晚期，却损失非候选区域。
g=1.05 的拟合期全局收益 0.00012167，选择期变为 -0.00008232。

在拟合期 g=1，共享强度的候选早期/晚期全局梯度贡献分别约 +1.142e-5、+1.919e-5，
非候选为 -5.435e-5；非候选占 92.61% 有效单元，总梯度倾向增强注入。
独立样本-节点坐标的三个区域两两余弦仍为正（约 0.455/0.053/0.102），
不能据此断言共享节点 MLP 的参数冲突。v12e 针对性比较损失权重，并测量实际 MLP 参数梯度。

## 冻结比较

协议 SHA-256：`2c0ca289ea3da9a6b8b6eb2e9487731f849243ef6330e6be9284ec7a3fd1e364`。
读取固定 v12c 协议继承所有时间资格、三种子、12 epoch、batch 8、Adam lr 0.001、
裁剪上限 5、恒等正则 0.001、评估 batch 16 和原选择规则。

两个版本均为原 v12c node 门控（100→16→1 Tanh，1,633 参数），保留原 backbone、
LayerNorm、TIID、动态图和预测器。全骨干冻结/eval，初始化 g=1 精确复现 A。

| 版本 | 拟合 minibatch 的数据损失 | 额外正则 |
| --- | --- | --- |
| global | 原全有效单元标准化 MAE；复用 v12c 原训练函数 | 原候选门控恒等正则 |
| regional | 候选 H1–H6、候选 H7–H12、非候选全 horizons 各自平均 MAE 的等权平均 | 相同正则及系数 |

每个区域在每个 minibatch 必须有有效单元，否则失败，不能静默改分母或重分配权重。
两个版本每种子从头重跑，使用相同初始门控状态及同样的逐 epoch 样本排列。
初始化哈希必须相等；不将既有 v12c scalar 当成这里的 global 节点版本。
global 版本继续调用原训练函数；区域版本仅改变数据损失聚合方式。

拟合期 W01–16 的 1,520 个完整事故样本用于优化；选择期 W18–25 为 856 个完整事故、
206 组共同三元组；审计 W27–35 为 1,010 个完整事故、378 组三元组。
W17/W26 间隔和原完整 support 筛选均继承。只有拟合事故 Y 进入优化器。

**选择目标仍是完整事故全局 MAE 严格改善**，并满足四 cohort、原受保护区域相对 A
不超过 0.1% 退化的点约束；epoch 0 始终可选。局部改善但全局无改善仍回退到 A。
不按 regional loss、梯度方向、审计结果或最好种子改规则。

## 参数梯度诊断

从合格拟合样本的有序索引中按 linspace 等位置选 64 条（不足则全取），仅依据 manifest。
两个版本和全部状态使用相同样本。分别在初始化、最后 epoch、最终选中状态计算实际
门控全部参数的区域损失梯度及恒等正则梯度，无 optimizer step。诊断前后保护随机数状态，
保证 DataLoader 等诊断操作不会扰动训练。

汇总该固定子集的各区域误差和/有效单元数，得到固定参数下的 pooled 区域均值梯度。
以子集有效单元占比组合 global 数据梯度；以 1/3 组合 regional 数据梯度；分别加入
0.001 正则梯度。保存参数名称、形状、按名称排列的向量、样本 ID、损失、范数和余弦。
正的区域-目标梯度内积意味着沿该目标的负梯度做无穷小更新时，该区域损失局部下降。

这不是全拟合期梯度、也不是 Adam 动量/预条件后的真实更新，更不能推出泛化改善。
初始化时隐藏层梯度可因末层零初始化而为零；不可误判为训练故障。
训练日志中的在线损失也不等于固定参数的拟合评估。

## 报告与信息边界

保留每轮历史、当前优化器与历史最佳状态、最终门控、三状态参数梯度、每样本区域误差。
审计报告每个种子的 global 相对 A、regional 相对 A、regional 相对 global 的收益，沿用共享周与四周块
bootstrap；全部 cohort/区域保存在 summary 中。保护规则是点估计筛选，不是风险保证。

旧 val/test 数组不读。原 A 和 scaler 曾使用整个 train，且后段审计已在 v12c 被查看；
此次是明确的重复开发比较，不是独立验证。不能将同一个审计时段包装成新 holdout。
任何下一步训练或测试集使用仍须依据全部结果另行决定。

## 服务器运行与恢复

在已分配 GPU（建议至少 3 CPU，沿用 igstgnn 环境）执行前台任务：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_regional_gate_objective.sh run contra_v12e_regional_gate_01
```

自动运行测试和真实 checkpoint 小样本两轮检查，然后执行六组正式拟合。运行中不拉代码、
不按 Ctrl+C；可以另开终端查看：

```bash
bash experiments/chronological/run_regional_gate_objective.sh status contra_v12e_regional_gate_01
bash experiments/chronological/run_regional_gate_objective.sh report contra_v12e_regional_gate_01
```

异常终止后保留原目录，确认原进程结束，在有效 GPU 分配内恢复到新运行名：

```bash
bash experiments/chronological/run_regional_gate_objective.sh resume contra_v12e_regional_gate_02 contra_v12e_regional_gate_01
```

源目录只读，核验代码/协议/输入/样本身份及 arm 对应训练设置，按完整 epoch 恢复。
已完成拟合不重新优化；A 参考、梯度诊断及最终审计重新计算。不同 arm 的恢复状态互不通用。
无最终 summary 时 report 明确显示未完成。代码有实质变化时应新开实验，不能绕过身份核验。
本地只有合成模型测试；真实 A/GPU 验证由服务器启动器执行，尚无 v12e 科学结果。

2026-09-29 本地验证：7 项 v12e 测试及 33 项原门控/诊断/恢复回归，共 40 项通过。
覆盖参数梯度有限差分、区域均值与全局均值的区别、空区域拒绝、初始化配对、诊断随机数
隔离、工程模式边界、跨 arm 恢复拒绝、恢复与不中断权重一致，以及启动器失败码传递。
Bash 语法、CLI、Python 编译及 diff 空白检查通过。
