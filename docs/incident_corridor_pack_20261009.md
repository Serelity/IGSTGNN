# M1.1 多站候选走廊输入包

2026-10-09 完成。任务见[模块开发清单](incident_capacity_propagation_tasks.md)。已交付训练历史输入包，不代表物理路网、量纲或预测效果已经通过。

## 真实结果

固定已有三对锚点，每端扩展 3 个源 postmile 单位，按配置中的候选方向保留范围内主线站。没有根据交通预测结果裁剪或排序候选。

| 走廊 | 主线站数 | 候选相邻区间 | 覆盖 postmile 跨度 | 报告关联训练窗口 | 含已知非主线点的区间 |
|---|---:|---:|---:|---:|---:|
| SR4-W | 16 | 15 | 5.526 | 1,116 | 9 |
| SR24-W | 15 | 14 | 5.250 | 309 | 2 |
| SR242-N | 9 | 8 | 2.480 | 214 | 5 |

跨度是源里程字段差，不是已核实的沿道路物理长度。报告关联沿用原 D 的候选支持，不是这些区间内实际事故数或真实受影响标签。三条走廊保留全部 3,604 个训练窗口，包括报告不关联该走廊的窗口；上表不参与筛掉负例。

共 40 站，数组 `[3604,12,40,3]`，包含流量、占有率、速度的源数值。12 步历史去重得到 28,320 个名义时间标签；重叠站点时刻三通道逐值一致。数值掩码采用有限且非负、保留零值的规则，三个通道均全为数值可用，**不意味着没有源插补或每条记录都是直接传感器观测**。

SR242-N 的 401393 和 401401 同为一个源里程值。两站均保留，候选区间标注 `coincident_postmile`，物理长度保持空值；后续需核实测量范围和道路分支，不能把它们作为零长度 CTM cell。发现的匝道/连接点保留类型、站号和里程，未取得其事故时刻动态流量。

## 文件与读取

入口：`experiments/chronological/prepare_incident_corridors.py`。固定选择规则：`experiments/chronological/incident_corridor_selection_v1.json`。读取接口：`src/utils/incident_corridor.py`。

| 输出 | 内容 |
|---|---|
| `train_history.npy` | 未标准化、未新插补的三通道训练历史，保留原无效值 |
| `train_value_usable.npy` | 同形布尔数值可用掩码，不冒称原始观测掩码 |
| `train_report.npz` | 原候选关联、报告年龄、预测时钟、样本及站点 ID |
| `train_rows.csv` | 样本/事故 ID、历史范围、报告时间、预测时点；不含事故描述或 duration |
| `stations.csv` | 原模型列号、打包列号、道路方向、源里程及坐标 |
| `corridors.json` | 各走廊站序及列号、候选连接、已知非主线点、待核验标志 |
| `selection.json` | 本次固定选择规则及原包指纹 |
| `summary.json` | 输入/输出哈希、范围、计数、语义、未解决的物理项 |

```python
from src.utils.incident_corridor import CorridorHistory

pack = CorridorHistory('/path/to/corridors')
window = pack.window(0, 'sr4_w')
# history_source_units: [12, 16, 3]
# value_usable: same shape; report_distances: [16, 3]
# station_ids: candidate travel order; physics_ready is False
```

读取器校验输出哈希和样本/站点映射；构建器核验已完成 v11a 与冻结原包的关联及源元数据。构建只打开白名单训练文件，不打开原始 `*_flow.npy`、验证数组或测试文件。读取的原 summary 可包含验证统计元数据，但不据其选择或比较走廊。

时间保持原 nominal 时钟：历史标签 t0−65…t0−10 分钟，报告在 t0 可用。源区间标签依据发布者解释为起点，历史观测与目标间有 10 分钟缺口。没有将报告回填为历史早期已知信息，初始报告快照与在线时延仍未认证。

构建先写独立 `.partial` 目录，读取核验后才发布正式目录。已有正式或残留 partial 目录均拒绝覆盖。阵列保留在运行目录，不提交到代码仓库。[机器证据](incident_corridor_pack_20261009.json)保存哈希、完整候选站序/边界和验证记录。

## 服务器入口

在更新代码后的仓库目录、已有 Slurm 资源内运行：

```bash
bash experiments/chronological/run_incident_corridors.sh
```

CPU 即可，建议 3 核、8 GB、15 分钟作为 I/O 和缓存恢复预留，不是服务器实测耗时。脚本自动使用 `igstgnn`；不包含 Git、下载、安装或提交调度任务的命令。

优先在原数据目录/同级目录、实验目录和已有研究目录的有限层级中寻找完整且匹配的 v11a 历史包；如多份包历史内容不同则停止并要求明确路径。未找到包时，尝试用原数据目录已有 `row_cache/blobs` 运行原 v11a 缓存物化器。此准备步骤会生成 train/val 历史，实际走廊选择与导出始终只使用 train X；不读取验证目标用于选择。缺缓存时明确报错，不自行联网找数据。

若 v11a 包放在其他位置，可直接给一个路径：

```bash
bash experiments/chronological/run_incident_corridors.sh /实际路径/v11a历史目录
```

可选环境变量沿用 `PHYSICS_DATA_DIR`、`PHYSICS_SENSORS`、`PHYSICS_PYTHON`，新增 `PHYSICS_HISTORY_DIR`。输出到新的 `experiments/chronological_runs/contra_incident_corridors_时间_随机后缀/`，包含日志、汇总和 `corridors/`。

预期 `CORRIDOR_TRAIN_HISTORY_PACK_COMPLETE`、`samples=3604`、`packed_stations=40`、`physics_ready=false`、退出码 0。`physics_ready=false` 表示 M1.2/1.3 的道路、观测及量纲工作尚待完成，不是本数据打包任务失败。

## 验证及范围

14 项新测试全部通过，覆盖身份/轴错位、跨切分/未来历史、重复样本、重叠冲突、NaN/负值/零值、匝道/同里程标记、部分写入失败、文件篡改、历史包查找和 Bash 参数/语法。Windows 上首次 Bash 启动被沙箱的 MSYS 命名对象限制阻止；在获准的本地执行环境下完成同一脚本全流程，退出码 0，没有修改测试或绕过数据校验。

真实 Bash 运行在 `contra_incident_corridors_20261009_162325_d0YE6U`，自动复用现有 v11a。独立按原站点列号逐站核对 **5,189,760 个数值单元及掩码**，全部与源训练历史一致；代码哈希与报告一致。数据包约 26.4 MB。缓存物化回退复用原 v11a 工具，本轮没有再次执行该回退路径；未提交服务器作业、训练模型或评估收益。

下一步处理 40 站的物理连接/观测映射及同里程站，不将候选里程图直接传入 CTM。
