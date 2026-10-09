# Chattanooga 作者数据来源与处理链溯源

核验日期：2026-10-09。已从美国能源部 OSTI 下载并解析两篇正式论文，核对首页标题、作者与 DOI：数据论文 12 页，事故检测论文 20 页。重点核读数据、融合、筛选和实验准备部分；未运行作者代码。原文与证据缓存位于本地 `论文学习/Chattanooga来源溯源_20261009/`。

文件身份、版本、元数据与本地补充统计见 [机器证据汇总](chattanooga_source_trace_evidence.json)。

## 原始数据与发布数据的关系

| 层次 | 查证结果 | 原文位置 |
| --- | --- | --- |
| 原始交通观测 | TDOT 的 Radar Detection System（RDS），逐车道、30 秒观测；研究涉及 Wavetronix SmartSensor V，属于 SmartWay 系统 | ESWA §3.2，PDF 第 4 页；DiB 第 6 页 |
| 原始事故 | TDOT E-TRIMS，道路执法记录，经 TDOT 清理、去标识后提供；报告时间来自 911 呼叫 | DiB 第 2、4—6 页；ESWA §3.1 |
| 气象 | NASA POWER，作者派生天气类别 | 两篇数据部分 |
| 光照 | Sunrise-Sunset.org，作者派生日光/暮光类别并用于 AM/PM 校正 | 两篇数据部分 |
| 我们下载的文件 | 上述来源经作者聚合、匹配和截窗后的 Zenodo 发布包；不是逐车道原始档案 | DiB 第 10 页；ESWA §3.6—3.8 |

本研究的道路是田纳西 Chattanooga 周边 I-75、I-24 等，时间为 2020 年 11 月至 2021 年 4 月。与我们 Contra 2023 年 PeMS 主任务的来源和地区不同。

**跨车道规则已有明确文字依据：车辆数求和，速度和占有率取各车道均值。** 出处为 DiB 第 10 页。这里确认的是作者声明的处理规则，不是异常单元格已经与逐车道原值逐一对应；尚未拿到执行该转换的代码或同一时刻的原始输入。

因此，之前对发布文件的极大数值检查仍成立。当前不能认定异常来自传感器、格式解析或聚合实现中的哪一环，也不能用未验证的缩放、拆数字或裁剪恢复真实车辆数。

## 原始数据能否直接取得

1. **Zenodo 发布包已取得。** 官方版本 API 当前只列出 v1.0（记录 7964288）；两个文件大小与 MD5 均与已下载副本一致。页面元数据更新时间不等于数据修正版，当前没有查到新增修正版本。
2. **E-TRIMS 原始入口要求登录。** [官方入口](https://e-trims.tdot.tn.gov/)显示 NET 域用户名和密码。DiB 伦理说明写明作者通过与数据所有者沟通获得使用权限；TDOT 数据所有者也是共同作者。不能把作者取得原始事故数据的方式描述成匿名开放下载。
3. **逐车道 RDS 历史档案未取得。** DiB 参考文献 [4] 只列出 TDOT Radar Detection Sensor Data，没有给出对应历史原始文件直链。查到 [TDOT 数据入口](https://www.tn.gov/tdot/long-range-planning-home/longrange-road-inventory-traffic.html)与 [TRIMS 请求入口](https://www.tn.gov/tdot/long-range-planning-home/longrange-roadway-data/longrange-road-inventory-trims-data-request.html)，但没有核验到可匿名下载这批 2020—2021 年、同站点逐车道 RDS 档案的地址。公开 AADT、实时事件或其他年份数据不能替代它。

此外，后期 [TDOT RDS 数据质量报告](https://rosap.ntl.bts.gov/view/dot/79431/dot_79431_DS1.pdf)§4.3 的公开检索内容描述合作服务器、实时数据馈送及不对外暴露的数据库。这是访问方式的补充背景，不能据此断言 2020—2021 年的档案绝不开放。未尝试任何非公开接口，未申请账户或发送数据请求。

## 找到的作者代码

通过官方 DataCite API，以精确论文题名及作者检索，定位到：

- **Supplementary code for: Spatiotemporal features of traffic help reduce automatic accident detection time**
- 作者：Pablo Moriano；发布年：2023；版本：1.0。
- DOI：[10.24433/CO.8627573.v1](https://doi.org/10.24433/CO.8627573.v1)
- 官方入口：[Code Ocean capsule 7228334](https://codeocean.com/capsule/7228334/tree/v1)
- 元数据说明其 Python 脚本用于生成论文图 6—17 对应结果。这个说明并不保证包含从逐车道 RDS 到 Zenodo 的原始转换代码。

本次 Code Ocean 页面返回 **HTTP 403**，没有下载或运行胶囊内容。不能写成“已经找到并审过聚合代码”。作者公开 GitHub 账户的仓库列表与精确仓库检索未定位到该论文仓库；其首页也未提供该论文 GitHub 链接。论文的 Code Ocean 复现标记支持进一步核对其运行环境和模型输入，但不单独证明每个发布数值物理正确。

## 作者主实验到底用了哪些样本

ESWA §3.8 报告主实验进一步按邻域与伤亡类型筛选：I-75 为 **24 起事故 + 3,039 个无事故事件**，I-24 为 **31 起事故 + 4,852 个无事故事件**。公开包覆盖的事件范围更广。

本地按 `bestData`、I-75/I-24、Fatal/Suspected Minor Injury/Suspected Serious Injury 筛选，事故数恰好得到 24 和 31。它支持“原论文使用更窄的伤亡事故子集”的解释，尚未认证逐个事件身份、对照选择或最终模型矩阵一致。

对这个**数量相符的候选子集**回查先前全窗口审查，15/55 个窗口含流量 >1,000，18/55 含占有率 >100。但这是完整约半小时文件的检查；论文特征截取范围是事件前 4 分钟至后 0—7 分钟。不能据此断言作者实际输入矩阵包含这些异常，更不能据此直接否定其事故检测结论。需要胶囊内的具体输入和转换实现才能进一步判断。

论文做的是事故检测，使用事件发生后逐渐增多的观测；我们的目标是事故信息辅助未来交通预测与物理状态建模，不能直接复用其输入截取和评估解释。

## 对上一轮审查的两项更正

### 车道和距离：已有元数据，尚需对应核验

DiB 表 9 明确 `SensorZones.geojson` 包含 `NBR_LANES` 与 `SPD_LMT`；本次直接检查得到 541 个多边形、177 个传感器键，车道数在 1—5 之间。这些键与拓扑 181 个键直接相交 142 个，29 个地理传感器键对应多个车道数，说明不能把每个多边形属性无条件作为唯一站点车道数。需要处理标识与路段对应关系，而不是说“没有车道信息”。

DiB 表 8 也明确 `PrevDist/NextDist/OppoDist` 单位为英里。上一轮“距离单位未确认”现已得到文字依据；实际物理路段边界、有效长度和车道测量配置仍需核对。

更重要的是，DiB 第 8—9 页说明岔路处只保留一个上游和一个下游，偏向同一道路或更大流量的方向。这表明发布拓扑用于邻居选择，可能省略真实分支流，不能直接当作完整守恒网络。

### 日期：正文解释了配对日期，文档内部仍需区分

DiB 表 1 将无事故 `date` 描述为观测日期，但表 10 及 ESWA §3.6—3.7 又说明复制事故属性，并保留原事故日期/时间。上一轮观测到的“CSV date 与无事故文件名日期差整周”与后一个说明一致。

因此，更准确的判断是：**应明确区分配对事故日期与实际观测日期，字典两处表述不一致；不能单凭这一差异认定数据损坏。** 先前禁止直接用 CSV date 划分观测日期的建议仍然适用。极大数值和共享观测的检查与这项语义更正相互独立。

## 对本研究的实际结论

数据来源、作者声明的单位和跨车道规则已明确。仍缺的是：同站点同时间原始逐车道记录、实际生成脚本、胶囊最终输入与发布文件的对应关系。当前不能认证“异常已修复”，物理训练继续暂缓。

最小有效的后续证据应是：任意一处异常站点时刻的逐车道原值，以及生成该发布单元格的代码片段；比继续下载更多同版截窗文件更能解决当前问题。Code Ocean 是已定位的复现入口；若无法取得其中的处理实现，则需要作者/TDOT 提供原始记录或转换说明。这里只记录所需材料，未联系任何人。

## 论文与证据清单

| 论文 | 元数据与获取状态 |
| --- | --- |
| Andy Berres, Pablo Moriano, Haowen Xu, Sarah Tennille, Lee Smith, Jonathan Storey, Jibonananda Sanyal. **A traffic accident dataset for Chattanooga, Tennessee.** Data in Brief 55 (2024): 110675 | DOI `10.1016/j.dib.2024.110675`；[OSTI 正式 PDF](https://www.osti.gov/servlets/purl/2397469)已下载、解析 12 页并核对身份；论文 CC BY-NC；数据包许可单独为 CC BY 4.0；Crossref 引用数 3（2026-10-09） |
| Pablo Moriano, Andy Berres, Haowen Xu, Jibonananda Sanyal. **Spatiotemporal features of traffic help reduce automatic accident detection time.** Expert Systems with Applications 244 (2024): 122813 | DOI `10.1016/j.eswa.2023.122813`；[OSTI 正式 PDF](https://www.osti.gov/servlets/purl/2251639)已下载、解析 20 页并核对身份；CC BY；2023-12 在线发表、2024 卷年；Crossref 引用数 21（2026-10-09）；代码入口见上 |

本地保存原 PDF、提取文本、Crossref/DataCite/Zenodo API 响应、文件 SHA256、地理属性统计、候选事故子集统计和检索/访问记录。没有重复下载数据压缩包、没有覆盖审查输出、没有训练或服务器提交。当前 Contra 主实验和 `PHYSICAL_CONTRACT_REQUIRED` 状态不变。
