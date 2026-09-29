# v12d：固定强度响应与拟合期区域梯度

## 来自 v12c 的依据

2026-09-29 用户回传轨迹诊断。六组训练每轮均更新参数且通过保护；五组从未改善选择期
全局 MAE。梯度最大值约 0.00202，小于裁剪阈值 5。scalar 学到略大于 1 的强度；
六组最后一轮均呈现候选区域损失、非候选区域收益的组合。node/2027 选择第 11 轮，
选择期全局收益仅约 0.00002117。这支持先检查强度方向的响应空间和区域收支，
不能据此认定正则主导、梯度冲突、学习率不足或必须增加模型容量。

## 冻结设计

协议为 `icsf_strength_response_v12d.json`，SHA-256：
`2d78447df846b6fd4e7b3cf631362611b6309f3120615ee913ad39d7e598070f`。
在运行任何真实响应曲线前固定以下规则：

- 使用原 A checkpoint，沿用 v12c 拟合 W01–W16、选择 W18–W25 及完整 support 资格。
  拟合仅完整事故 1,520 条；选择完整事故 856 条和 206 组共同三元组。
  不取用 W27–W35 后段审计样本，不读 val/test 数组。
- 强度网格为 0、0.5、0.8、0.95、0.99、1、1.01、1.05、1.2、1.5、2。
  在每个 cohort 全部报告，无最佳强度排名或新 checkpoint 选择。
- 只替换 ICSF 注入强度；保留 LayerNorm、TIID 原上下文及动态建图/预测计算。
  g=0 不等于 incident_off，也不等于禁用 TIID；g=2 是有限 sigmoid logit 无法达到的诊断端点。
  g=1 在所有评估批次上必须逐元素复现原模型预测。
- 全模型 eval，参数冻结，无优化器。扫描 batch=16，区域梯度 batch=8，线程=3。
  GPU 用于前向计算和 autograd；这是梯度诊断，不是训练。

沿用的输入核验会哈希完整 train 文件，数据包装器会加载原 train 元数据及上下文；
上述边界指仅索引拟合/选择期样本进行预测和目标评估，并非不接触包含后段记录的文件字节。
原 A、scaler 和开发期均已被使用过，本轮不能提供独立泛化证据。

## 梯度的确切含义

仅在拟合期完整事故、g=1 上，为每个样本和站点建立独立的直接强度坐标。
对候选 H1–H6、候选 H7–H12、非候选全部 horizons 三个互斥区域的标准化绝对误差和求导。
实现以扫描的 raw 预测误差除以冻结 flow std，和标准化 MAE 数学等价，但浮点舍入顺序
可能与 v12c 训练损失稍有差别。

保存各区域原始导数与有效单元数。各区域除以共同的完整拟合期有效单元数后，得到可加的
全局目标梯度贡献。另报告按各区域自身单元数归一化的平均误差导数，避免混用分母。
独立求出的整体梯度必须等于三部分之和（浮点容差内）。

将所有样本、节点坐标导数求和，得到共享 scalar 强度的局部导数；乘 0.5 得到 v12c
在 g=1 时的 scalar logit 导数。正导数意味着局部减小 g 会下降该损失。
报告区域坐标向量余弦、整体向量范数除以区域范数和，以及共享 scalar 的区域抵消比；
比值越低表示在该坐标系下抵消越强，零分母报告 null。

这些不是共享节点 MLP 的参数梯度，也不是 Adam 每批更新轨迹。独立坐标的有益方向可能
无法由报告时可见特征预测。g=1 的恒等正则导数恰为零，无法据此判断训练后的正则主导性。
额外对照固定 0.99/1.01 的拟合期 raw MAE 割线斜率与局部导数；MAE、建图非光滑与
浮点舍入均可使二者不同，不能自动把这种差异判定为实现错误。

## 产物与服务器运行

产物：冻结协议及代码/输入哈希、完整资格与实际索引、各点每样本区域误差与 counts、
response.csv、带原样本 ID 和 station ID 的 fit_strength_gradients.npz、summary.json。
报告均为点估计，不新增置信门槛、训练选择或自动的下一阶段许可。有限网格不能证明最优点。
失败保留 .partial，成功才发布最终目录；此轮无断点续扫，重跑使用新运行名。

在已经分配的 GPU 环境（建议至少 3 CPU，沿用原 igstgnn 环境）执行：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code || exit 1
git pull --ff-only origin research/chronological-tiid
conda activate igstgnn
bash experiments/chronological/run_icsf_strength_response.sh run contra_v12d_strength_response_01
```

run 是前台入口，可用于平台批任务命令；会先运行测试和两条样本的小检查，再运行全量。
不要在前台 run 时按 Ctrl+C；另开终端查看日志或状态：

```bash
bash experiments/chronological/run_icsf_strength_response.sh status contra_v12d_strength_response_01
bash experiments/chronological/run_icsf_strength_response.sh report contra_v12d_strength_response_01
```

本地无真实 A checkpoint，科学结果需服务器回传。2026-09-29 本地 9 项测试通过，
Bash 语法、CLI 和 diff 空白检查通过。小模型合成测试覆盖恒等复现、
端点/TIID 上下文保护、自动求导与有限差分、梯度分区与批次划分一致性、未连接坐标零梯度、
不取用后段样本、不执行 optimizer、输入失败记录和前台失败码传递。
