# H1 物理数据溯源与瓶颈准备

日期：2026-10-09；接续服务器工程检查 1163672。已完成公开来源核读、历史车道元数据获取、训练 X 去重统计、匝道与重合站筛查，以及可选官方源文件逐值比对工具。仍未形成可用于真实物理训练的完整契约。

最新服务器结果：作业 **1163917** 已完成新准备入口，14 项测试通过、退出码 0，打印的文件身份和非路径汇总与本地一致。原始 PeMS 比对尚未执行，无需重复无参数准备；详见[服务器结果解读](incident_physics_evidence_server_check.md)。

## 当前证据

| 项目 | 本次得到的依据 | 当前判断 |
|---|---|---|
| 时间标签 | TraffiDent 正式论文 §4.1.1 明确将 15:40 对应到 15:40–15:45 | 发布者定义为区间起点；时区、DST、延迟仍未独立认证 |
| 流量单位 | 作者下载脚本指向 PeMS Station 5-Minute；本地构建只取原数组第 0 通道；训练 X 有效数值均为整数 | 支持五分钟计数的解释，但缺官方原记录→v8 数组的逐值对应，尚不自动填写单位 |
| 车道数 | 新取得 LargeST v1 的 `ca_meta.csv`，8600 行、570431 字节 | 488/496 个 ID 匹配，487 个连道路、方向、类型、坐标也一致；不是 2023 年车道认证 |
| 观测边界 | 完整源元数据含当前模型包未纳入的匝道/连接检测器 | 482 对里程候选中 261 对区间内有已知非主线点；217 对通过元数据预筛，0 对获封闭物理边界认证 |
| 容量与队列尺度 | 已构建去重训练 X 统计 | 未做容量拟合；分位数不自动视为容量，尚未填入车辆数尺度 |

来源等级必须分开：发布者直接描述、代码事实、跨数据集对应、数值提示、未解决问题。`PHYSICAL_CONTRACT_REQUIRED` 继续保留，不能用完整性检查替代物理事实。

## 时间窗口的具体含义

按区间起点解释，现有冻结标签不变：

```text
X 末标签 T−10  →  最后历史区间 [T−10, T−5)
缺口           →  [T−5, T+5)，共 10 分钟
Y 首标签 T+5   →  第一个目标区间 [T+5, T+10)
Y 末标签 T+60  →  最后一个目标区间 [T+60, T+65)
```

新准备器将 `interval_label=start` 和证据出处写入未完成草案，仍为 `status=unresolved`。这不改原 X/Y 切片、预测时钟、历史指标或测试资格。基础 H1 工程审计不会自动导入这份新证据，因此没有草案输入的旧入口仍会列出四类待核依据。

## 得到的新数据与来源边界

LargeST 的官方仓库公开说明其元数据含车道数，原始交通时期为 2017–2021。仅下载了 Kaggle 数据集 `liuxu77/largest` v1 中的小型 `ca_meta.csv`，没有下载年度交通数组，也没有将不同年份交通作为本研究训练数据。

文件 SHA-256：`57c5edb5e6f1d1802440426f2bd07f54d812859135989c740ab96b1448b087ee`。

8 个未匹配 ID：408608、402071、405931、401401、427183、402540、408621、402070。400774 的 ID 匹配，但道路/方向/类型/坐标组合核对未通过，本次检查定位到坐标差异。所有匹配结果均单独记录，不用历史车道数覆盖当前源表，不用道路总宽除以车道宽替代实际车道数。

作者公开仓库的主提交 `b773335d31fa6e027026d5167489cbab7c7061e9` 已检查文件树，排除其中附带的第三方环境目录。`fetch_url_list.py` 提供 2022 年 Station 5-Minute 下载列表入口；所读 causal preprocessing notebook 已从月数组开始处理。此次未找到把 2023 年 PeMS 原始列转换成 v8 月数组的完整代码。论文附录中概括性的“通常车辆/小时”不能替代这个转换证据。

PeMS 官方历史档案入口要求账户。本次匿名公开访问未取得所需的 2023 年 District 4 Station 5-Minute 原文件；没有因公开代码提到登录就假定已拥有下载权限。取得官方导出记录或可核验的转换脚本后，才能继续闭合 v8 单位证据。

## 元数据预筛与训练支持

完整原始表中 Contra Costa 有 773 个点：496 个 Mainline、153 个 On Ramp、113 个 Off Ramp、11 个 Fwy-Fwy。当前模型包只包含 Mainline。

准备器逐路、逐方向检查相邻主线里程对，并包含端点：区间内已知匝道/连接点、相同里程、端点里程的其他主线点均记为待排除问题。因此有 217 对通过这一元数据预筛；它们只是后续地图、道路交换和观测算子核验的候选，不是 217 个瓶颈。未知或未布设检测器的匝道仍可能存在。

预筛通过的分布：I580-E/W 为 7/8，I680-N/S 为 29/37，I80-E/W 为 11/15，SR24-E/W 为 18/17，SR242-N/S 为 3/1，SR4-E/W 为 36/35。没有使用验证目标、预测误差或模型收益来选路段，也没有把里程差解释为已认证的 cell 长度。

训练包共有 3604×12=43248 个窗口历史时间槽，去重后为 28320 个名义时间标签，移除 14928 次重复。每次重复都核对全部站点的值，真实零和缺失分别保留；同一站/时间发生不一致立即失败。数值统计与报告支持计数只使用训练 X/context。包完整性校验会读取文件字节计算哈希，但不将验证、gap 或 Y 的值用于统计。

到达站到服务边界还存在行驶时延，两个空间检测器的流量差也包含行驶中车辆数量变化，不能直接把它解释为排队车辆增长。后续必须给出到达延迟/观测算子及侧向流处理，再确定容量与队列尺度。第一版可先研究局部瓶颈，但不以“距离短”跳过这些定义。

## 工具、产物及验证

Python 入口：`experiments/chronological/prepare_incident_physics_evidence.py`。必需参数为 `--data-dir`、`--published-sensors`、`--output-dir`；元数据选择 `--raw-sensors`（可搭配 `--historical-lanes`），或 `--metadata-bundle`。两种方式不能混用。服务器使用下述 Bash 即可。

产物：

- `unique_train_x_station_profiles.csv`：去重后的有效数、零值、分位数，保留 source units。
- `bottleneck_candidate_inventory.csv`：全部 482 对候选及非主线点、重合点、数据可用率和训练报告支持。
- `historical_lane_crosscheck.csv`：历史车道对应及未认证标记。
- `contract_draft_UNRESOLVED.json`：只补入发布者声明的时间标签，不伪填单位、瓶颈或容量。
- `source_record_request.json`、`summary.json`：源记录比对需求与可复核汇总。

可选 `--pems-5min <官方CSV/txt或gz>` 会按站号和时间标签，只对比训练 X 的有效单元，区分“原五分钟计数”与“乘 12 后的小时率”。要求至少 100 个唯一有效单元、2 站、2 个时间标签且所有值匹配，拒绝冲突重复；全零、错位、混合或其他缩放不会自动确定单位。匹配本身不认证文件来源，结果不自动启用训练或填写物理契约。

9 项新增测试通过，覆盖 X 去重、重复冲突、未来槽隔离、缺失值、非训练期拒绝、匝道端点、重合主线点、历史元数据错位、计数/小时率区分和全零歧义。与既有 H1 测试一起运行，共 30 项通过、无跳过（Python 3.10.18、PyTorch 2.3.1 CPU）。已在本地真实 v8 数据上运行最终版准备器；可复核的计数与文件身份见[本次汇总](incident_physics_evidence_20261009.json)。官方源记录的比对目前只有合成测试，尚未进行真实 PeMS 文件比对。

无需再提交 GPU 作业来重复这一步。模型与训练器没有修改，真实物理窗口更新尚未开始。后续顺序为：取得可验证的原记录或转换代码，核实流量单位与车道聚合；从候选表中核实少量路段的实际边界与观测时延；再做训练期容量/队列尺度校准，接入完整 fixed、同输入 GRU、点队列三臂对照。

## 服务器入口（后续补充）

用户要求服务器一键执行后，新增 `run_incident_physics_evidence.sh`，把上述准备和源记录比对接到同一个入口。它只使用 CPU，沿用已通过 H1 检查的 `igstgnn` 环境及原 v8 train/val 包。

先在服务器仓库目录单独更新代码：

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
git switch research/chronological-tiid
git pull --ff-only origin research/chronological-tiid
```

随后在已有 Slurm 资源分配中运行，或将原作业脚本中的执行命令替换为：

```bash
bash experiments/chronological/run_incident_physics_evidence.sh
```

脚本内部没有 Git、下载、安装或调度命令。建议 CPU 3 核、内存 8 GB、15 分钟作准备/I/O 预留，无需申请 GPU。默认路径沿用 H1 的 `../data/chronological/Contra_Costa_v8_dev` 与 `../data/xtraffic/Contra_Costa/sensors.csv`，可设置 `PHYSICS_DATA_DIR`、`PHYSICS_SENSORS`、`PHYSICS_PYTHON` 覆盖。

必要元数据摘录已随代码附带：保留候选道路/方向所有县的源站点及 Contra Costa 全部站点（2217 行），以及匹配站号的历史车道表（488 行）。来源、摘录规则、原文件和摘录文件哈希、CC BY-NC 4.0 署名见 [physics_metadata/README](../experiments/chronological/physics_metadata/README.md)。这不是更换交通或事故数据，车道数也没有加入训练输入。

新入口先跑 14 项证据/启动器测试，再核验元数据哈希、准备各表并输出源记录请求。完整 H1 相关测试当前共 35 项全部通过、无跳过。本地已完整执行 Bash，退出码 0；使用完整源表和随附摘录生成的三张结果 CSV 逐字节一致。随后服务器作业 1163917 也已回传成功，14 项测试通过；收到的是控制台日志，服务器原始 CSV 尚未传回，真实官方 PeMS 原文件尚未比对。

输出目录为 `experiments/chronological_runs/contra_physics_evidence_时间_随机后缀/`。请回传 `run.log` 末尾 JSON 或 `summary.json`。预期为：

```text
status: PHYSICS_EVIDENCE_PREPARATION_COMPLETE
readiness: PHYSICAL_CONTRACT_REQUIRED
unique_train_x_nominal_slots: 28320
metadata_prefilter_pass_pairs: 217
historical_lane_road_coordinate_matches: 487
Workflow exit code: 0
```

`evidence/source_record_request.json` 现在列出具体日期、训练 X 时间范围、站号和匹配覆盖量。只按训练 X 非零/有效覆盖量排序；当前前三日为 **2023-07-10、2023-02-27、2023-06-29**。可先从有权访问的 PeMS 账户取得 District 4、2023-07-10 的 **Station 5-Minute 原始日文件**，保留全部原始列。其余日期是备选，不是要求一次下载三日。

取得文件并放到服务器后，在相同目录执行：

```bash
bash experiments/chronological/run_incident_physics_evidence.sh /实际路径/官方原始日文件.txt.gz
```

支持未压缩 CSV/txt 或 gzip；不要求按示例重命名。新日志的 `optional_source_record_comparison` 报告匹配量及单位候选。脚本不会代登录 PeMS，也不会凭元数据或样例日期自动补齐单位。运行成功表示准备完成；即便单位匹配，真实边界、到达时延及容量/队列尺度仍须解决后才能推进训练。

## 原始来源

沿用 academic-search 的原始来源核验方法；此次是数据契约核验，没有新增效果复现。

1. Gou, X.; Li, Z.; Lan, T.; Lin, J.; Li, Z.; Zhao, B.; Zhang, C.; Wang, D.; Zhang, X. (2025). *TraffiDent: A Dataset for Understanding the Interplay Between Traffic Dynamics and Incidents*. NeurIPS 38 Datasets and Benchmarks. DOI: 10.52202/085713-2796；arXiv:2407.11477。已在线核读[正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/7813e19a86fd73d40f7e811ab15f6d5f-Paper-Datasets_and_Benchmarks_Track.pdf) §4.1.1、A.8、A.10 相关段落，未逐页精读；引用数未查询。[作者代码](https://github.com/XAITraffic/XTraffic)。
2. [作者下载列表脚本](https://github.com/XAITraffic/XTraffic/blob/b773335d31fa6e027026d5167489cbab7c7061e9/process/fetch_url_list.py)及[因果分析预处理 notebook](https://github.com/XAITraffic/XTraffic/blob/b773335d31fa6e027026d5167489cbab7c7061e9/causal_analysis/globalcausal/data_preprocess.ipynb)：仅核查数据来源与转换范围，未执行外部代码。
3. [Caltrans / California 数据变换说明](https://cagov.github.io/caldata-mdsa-caltrans-pems/data/data-transform/)：五分钟车道/站级聚合背景；现代化系统文档不独立证明 v8 数组的具体转换。
4. [CPP 交通工程教学中的 PeMS 流量输入](https://www.cpptranspo.org/traffic-data-input)：五分钟总量到小时率乘 12 的数据使用说明；不用于认证 v8 年份和文件身份。
5. [LargeST 官方仓库](https://github.com/liuxu77/LargeST)及[官方 Kaggle 发布](https://www.kaggle.com/datasets/liuxu77/largest)：历史车道元数据的来源；实际下载版本 1、文件 `ca_meta.csv`，接收清单和哈希保存在本地研究目录。
6. [PeMS 官方入口](https://pems.dot.ca.gov/)：账户与历史档案访问入口，当前未取得登录后的源记录。
