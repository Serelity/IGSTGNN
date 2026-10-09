# 事故与交通物理研究：已下载的公开论文数据

核验日期：2026-10-09。用户要求寻找其他论文公开、可以直接下载的数据。此次实际下载两套、六个文件，共 123,329,674 字节（约 123.33 MB），全部与发布方的 MD5 和文件大小一致，并记录 SHA256。

下载文件保存在本机 `G:/paper/IGSTGNN_open_source/论文学习/替代公开数据检索_20261009/`。服务器尚未下载或导入这些文件。仓库只保存本说明和 [下载清单](incident_physics_alternative_downloads.json)，不收录数据压缩包。

## 结论与当前主实验的关系

**后续全量审查更新：** Chattanooga 已发现发布 CSV 中异常巨大的流量/占有率、对照窗口日期含义差异与大量共享观测。当前结论为暂不进入物理训练；详见 [2026-10-09 数据审查](chattanooga_data_audit_20261009.md)。以下“优先检查”是下载阶段的候选排序，不代表审查通过。

Chattanooga 是优先检查的补充数据：事故、上下游交通序列和传感器拓扑同时公开，物理量的采样单位也有作者说明。可据此研究事故附近短时传播，以及新增事故条件路径、物理状态路径是否有用。它尚不是可直接接入现有训练器的数据集。

PLOS ONE 公开包可作为真实 PeMS 文件格式及事故—检测器匹配的参考；实查其原始流量只有 2017-01-01、District 3，与当前 Contra 496 站没有 ID 交集，不能拿它证明现有 2023 年 v8 数组的单位或转换关系。

当前 IGSTGNN/Contra 主实验保持不变，H1 仍为 `PHYSICAL_CONTRACT_REQUIRED`。这次下载没有启动新训练，也没有解除当前流量单位、车道聚合、物理边界与容量校准的缺口。参见 [现有物理数据契约](incident_physics_data_contract.md)。

## 1. Chattanooga：事故和上下游交通序列

作者数据：[A Tagged Traffic Accident Dataset for Machine Learning](https://zenodo.org/records/7964288)，v1.0，2023-05-23；数据 DOI `10.5281/zenodo.7964288`；CC BY 4.0。

关联论文（数据发布页明确列出）：

- Pablo Moriano, Andreas Berres, Haowen Xu, Jibonananda Sanyal. **Spatiotemporal Features of Traffic Help Reduce Automatic Accident Detection Time.** *Expert Systems with Applications* 244 (2024): 122813. [DOI](https://doi.org/10.1016/j.eswa.2023.122813)。[作者 PDF 地址](https://pmoriano.com/docs/ESWA-24.pdf)本次未成功读取，不能记作全文已读。
- Andreas Berres, Pablo Moriano, Haowen Xu, Sarah Tennille, Lee Smith, Jonathan Storey, Jibonananda Sanyal. **A Traffic Accident Dataset for Chattanooga, Tennessee.** *Data in Brief* (2024): 110675. [DOI](https://doi.org/10.1016/j.dib.2024.110675)。本次依据数据页确认书目信息，未核读论文全文。

代码仓库未在本轮核验；引用次数未查询。

下载 `annotatedData.zip` 和 `metaData.zip`，共 109,158,682 字节。作者说明采样间隔为 30 秒，流量是该间隔内车辆计数、速度为 mph、占有率为百分比；序列围绕事故报告时间截取前后各约 15 分钟。报告时间来自 911 呼叫，不等同于事故真实发生时刻。

直接读取归档后得到以下清单；数量来自本地文件检查，不是对网页概述的估算：

| 文件或目录 | 实际内容 |
| --- | --- |
| `allData/accident` | 702 个事故窗口 CSV |
| `allData/non-accident` | 17,357 个无事故窗口 CSV |
| `bestData/accident` | 361 个事故窗口 CSV |
| `bestData/non-accident` | 8,968 个无事故窗口 CSV |
| `Accidents.csv` | 1,593 行事故元数据；不能视为 1,593 个可训练窗口 |
| `SensorTopology.csv` | 181 行；含前后站点、方向、距离、经纬度等字段 |
| 其他元数据 | `SensorZones.geojson`、天气和光照字典 |

按发布说明，`bestData` 是具备完整上下游各五个邻居的子集，不能与 `allData` 相加计数。实际目录名是单数 `accident` / `non-accident`，与网页文字中的复数拼写有差异。文件名日期范围为 2020-11-01 至 2021-04-29。

抽查 `bestData/accident/2020-12-09-1457-00I75S13.7.csv`：62 行、43 列，时间为 14:42:00—15:12:30；最后 33 列为事故附近 11 个站点的三种交通测量。该抽查不代表所有文件均无缺失、严格等长或完成了时间对齐审计。

### 对我们研究的启发及使用前提

作者按相同地点、星期及钟点匹配无事故窗口的做法，可启发我们为事故条件模块设计更有针对性的对照。但这种匹配不是因果效应认证，还需防止同一事件关联窗口或重叠观测跨训练/验证划分。

已有上下游顺序，可用来检查事故后上游积累、下游放行变化是否支持设计中的状态表达。不过拓扑邻接不证明区间没有匝道，也没有直接提供队列长度或容量真值。点队列仍需到达/离去观测映射；CTM 仍需距离单位、路段长度、边界流、车道汇总与参数依据。

只公开约半小时事件窗口，无法直接复用当前 60 分钟预测目标，也不能假定覆盖事故后的完整排空与恢复过程。若用于补充机制实验，应另行定义短时预测协议，并保持完整 fixed、同输入自由递推、物理分支之间的信息公平。事故类型及其他事后整理字段不能默认在首次报告时可用。

窗口缺失、时间对齐、拓扑关系及共享观测的全量审查现已完成，结果见上方审查报告。首次报告可用字段、异常数值来源、完整物理边界仍未认证；尚未开发训练数据包。

## 2. PeMS / PLOS ONE：事故风险研究公开包

Dongye Sun, Yunfei Ai, Yunhua Sun, Liping Zhao. **A highway crash risk assessment method based on traffic safety state division.** *PLOS ONE* 15(1) (2020): e0227609. [论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227609)，DOI `10.1371/journal.pone.0227609`。

论文直接链接 [Figshare 数据 v1](https://doi.org/10.6084/m9.figshare.10303868.v1)，CC BY 4.0。本次核读数据可用性及数据采集、匹配部分，未将其记为全文精读。代码仓库未核验；引用次数未查询。

四个公开文件共 14,170,992 字节：`the matched sample dataset.xlsx`、`detector dataset.xlsx`、`crash dataset.xlsx`、`traffic flow dataset.gz`。

本地直接检查结果：

- 流量 gzip：367,776 条记录，全部属于 2017-01-01、District 3，共 1,277 个站点，其中 617 个 ML 站点；每行 52 列，包含车道相关字段。
- 与当前 Contra 496 个站点的 ID 交集为 0。这是独立地区和年份的原始文件，不能作为当前 2023 年 D4 的逐值比较源。
- 检测器主工作表占用范围为 1,278 行、14 列；事故表为 44,040 行、20 列。这些是工作表尺寸，包含表头，不直接解释为有效记录数。
- 匹配样本有多个工作表；其事故风险用途主要使用事故前状态。公开单日原始流量是否足以重建所有匹配样本未验证，因此不把该包称为完整、连续的事故后预测数据。

研究借鉴是“先按时间和位置匹配事故与上下游观测，再建立有针对性的对照”。当前可用于格式与匹配方法参考；如需正式事故后预测，应先取得足够完整的连续流量。

## 3. 其他候选及检索边界

- **Incident congestion propagation prediction using incident reports**，SuMob 2023：[论文 PDF](https://www.ikg.uni-hannover.de/fileadmin/ikg/Service/SuMob23/6_SuMob_23_paper_7_10p.pdf)、[作者仓库](https://github.com/MathiasNT/CongestionPropagationPrediction)。本轮阅读数据节与 README；论文描述洛杉矶 2017 年 215 个检测器、1,024 起事故。核查的仓库树中未发现可直接下载的成套原始数组，README 对原始模拟数据写明需联系提供。仅保存 README 与仓库树快照，未取得这套数据，也未联系作者。
- **A hybrid model for missing traffic flow data imputation based on clustering and attention mechanism optimizing LSTM and AdaBoost**，*Scientific Reports* 2024：[论文](https://www.nature.com/articles/s41598-024-77748-1)。虽然使用 2023 年 PeMS 数据，数据声明为合理请求后向通讯作者获取；本次没有取得公开数据包。
- 本轮定向搜索了 PeMS 2023、District 4、`d04_text_station_5min_2023_07_10`、GitHub、Figshare、Zenodo 等组合，尚未找到经过核验、能与当前 v8 训练 X 直接逐值比较的 2023 D4 公开源文件。这是有限检索结果，不是证明不存在公开副本。

## 证据与复核

本地同目录保存发布 API 响应、下载清单、`audit_downloads.py` 和 `content_audit.json`。未批量解压数据、未运行下载包中的代码。MD5 与发布方比对确认下载完整性；SHA256 用于后续文件身份核对，不代表数据语义已得到独立认证。

下载清单给出每个文件的公开 URL、文件大小、发布方 MD5 与本地 SHA256。所有后续训练数据选择应沿用当前研究目标：验证事故信息与物理状态辅助路径能否改善完整 IGSTGNN，而不是以换数据或改变任务本身代替模块效果验证。
