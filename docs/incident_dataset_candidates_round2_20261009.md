# 其他论文的事故与交通观测数据：第二轮候选

2026-10-09。用户质疑不应只停留在 Chattanooga，随后提出按现有数据缺口查找。本轮继续使用 academic-search，检索事故辅助交通预测、事故影响预测、传感器异常检测三个方向；按字段可用性排序，未以下载次数或引用数代替数据质量。

## FT-AED：优先审查的独立机制数据

Austin Coursey, Junyi Ji, Marcos Quinones-Grueiro, William Barbour, Yuhang Zhang, Tyler Derr, Gautam Biswas, Daniel B. Work. **FT-AED: Benchmark Dataset for Early Freeway Traffic Anomalous Event Detection.** NeurIPS 37 (2024), 15526–15549. [arXiv:2406.15283](https://arxiv.org/abs/2406.15283)；[作者项目](https://acoursey3.github.io/ft-aed/)；[数据](https://github.com/acoursey3/freeway-anomaly-data)；[项目链接的检测代码](https://github.com/acoursey3/freeway-anomaly-detection)。

- Nashville 附近 I-24，49 个里程位置、4 车道、196 节点，逐车道30秒速度、车辆数和占有率。主包为2023年10月工作日上午04:00—12:00，作者报3,763,200点；非全天连续全年数据。
- 主论文说明42起官方报告事故、19个人工补充异常；人工异常不全部是事故。仓库另列11个月每月一周的补充数据与原始事件日志入口，完整内容尚未核验。
- 已读arXiv v2的数据与处理章节以及当前仓库。速度采用交通波形插补，流量/占有率局部平均，不能直接把处理后的序列认证为实时预测输入。事故报告存在延迟；无向基准图也不自动是守恒路网。
- 当前仓库 LICENSE 标题为 BSD 3-Clause，README 明确旧补充材料的CC BY-NC说法已调整。数据CSV使用Git LFS；取得的是134字节指针，声明真实文件123,234,268字节、SHA256 `9bde51bd3e57ebffec6145b16f6b3ea724394a748f1d8483c9485e8a6ab42490`。**真实CSV尚未下载，不把指针当数据。**
- 评价：数据字段与“事故＋物理”较贴合，值得优先审查，但事故数有限、插补和首次可用时点仍需审查；不能补当前Contra缺值。

阅读状态：HTML方法部分已核读，PDF未本地下载；NeurIPS PDF的web读取因16MB大小限制失败。正式会议信息经作者项目/仓库核对。引用数未查询。

## Incident-LA / Incident-SB：事故影响验证候选

Yanshen Sun, Kaiqun Fu, Chang-Tien Lu. **RoadFormer: Road-Anchored Adversarial Dynamic Graph Transformer for Unlimited-Range Traffic Incident Impact Prediction.** IEEE Big Data 2023, 895–904。[作者仓库](https://github.com/styxsys0927/RoadFormer)；[作者PDF](https://people.cs.vt.edu/~clu/Publication/2023/IEEE-BD-Sun-2023.pdf)。DOI本轮未核验，引用数未查询，PDF仅有检索内容与数据表证据，未完成全文精读。

仓库披露LA 5,668起、1,663站；SB 1,452起、1,150站。每个样本18步，包括事故前12步和后6步，交通特征为**速度、占有率**，另列事件属性、持续时间和标作队列长度的空间影响目标，道路/站点关系和距离。

公开README有Dropbox下载链接，但本轮工具未成功读取目录，数据未取得。适合进一步查事故时空影响标签的构造与传播验证；没有发布流量列的证据，不作为当前流量守恒模块的直接替代。事件后6步及duration/q_len不能作为首次报告时的输入；空间目标也不能未经方法核验称独立队列真值。

## EXPY-TKY：事故场景下的速度预测候选

Renhe Jiang, Zhaonan Wang, Jiawei Yong, Puneet Jeph, Quanjun Chen, Yasumasa Kobayashi, Xuan Song, Shintaro Fukushima, Toyotaro Suzumura. **Spatio-Temporal Meta-Graph Learning for Traffic Forecasting.** AAAI 37(7) (2023), 8078–8086。[arXiv:2211.14701](https://arxiv.org/abs/2211.14701)；[官方MegaCRN仓库](https://github.com/deepkashiwa20/MegaCRN)；[数据目录](https://github.com/deepkashiwa20/MegaCRN/tree/main/EXPYTKY)。引用数未查询，PDF本轮未下载，核对范围为摘要与作者仓库。

东京高速，论文采用1,843条路段，10分钟粒度、2021年10—12月。官方仓库有三个按月CSV.gz、路段属性、邻接文件；后续[TESTAM官方代码](https://github.com/HyunWookL/TESTAM)也提供使用该数据的入口。

本轮未读取完整CSV字段，不把第三方所称“含事故”直接认证为具有首次报告时间、封道更新或流量。已确认的核心任务是速度预测，因此优先级低于FT-AED的物理字段核验。任务可借鉴不等于数据可以直接拼接到当前Contra。

## 检索边界与结论

本轮发现新候选足以纠正“只有Chattanooga或必须等原数据”的过窄路径。**字段更贴合的候选已经找到，数值质量尚未认证。** 根据用户最新方向，先用[数据缺口清单](data_gap_inventory_20261009.md)补现有主线，新数据保留作独立验证选择，不立即更换实验目标。

实际搜索词包括：`traffic incident forecasting dataset flow speed accident open source github`；`traffic incident dataset upstream downstream volume speed zenodo figshare`；`incident-aware traffic dataset`；`traffic forecasting incident benchmark dataset`；`FT-AED dataset github`；`EXPY-TKY dataset download`；`Incident-LA dataset github RoadFormer`。第三方索引只用于发现，结论回查论文、作者项目/仓库。

没有联系作者或下载执行外部代码。FT-AED的Box补充目录、RoadFormer的Dropbox目录本轮web工具未读取成功；这些情况不证明资源不存在。主要候选已核实到论文与作者发布入口，均未完成全量质量审查。
