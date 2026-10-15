# 事故条件消息传播：数据与理论合同

_2026-10-15；承接模块职责重设计与理论审查。状态：设计合同已具体化，完整模型尚未实现。_

---

## 🎯 范围与决定

用户要求先补齐理论审查发现的欠定义。本轮交付数据事实、确定的算子公式、时间与图接口、条件性质及可运行的反例检查。它们不等于模型效果验证。论文主 baseline 仍是原 IGSTGNN。

本合同细化并部分替代[上一轮草图](igstgnn_module_redesign_after_f3_20261015.md)：报告不仅改变消息权重，也能在同一个 DIF 算子内改变消息内容；源位置保持可定位；明确硬支持与时间来源类型。F2/F3 的现有代码与历史制品继续保留。

| 决定 | 原因 | 代价 |
|---|---|---|
| 同时条件化权重与消息内容 | 纯权重在同质 value 下必然失效 | 增加内容映射，须区分两种作用 |
| source token 单独做 value 变换 | 保留新算子入口处的源位置 | 删除旧跨全窗非线性变换，不保证旧函数类包含 |
| 保留各图/阶与 self bank | 避免将多图表达压成一个凸平均 | 计算量仍需实测 |
| 使用固定邻接的硬支持 | 定义可核验零边 | 损失旧稠密图的部分远程依赖 |
| 来源类型与生成块序号分开 | 避免历史槽与潜块混为相同时间间隔 | 不提供物理时延或事故年龄 |

F2 同检查点融合 MAE pooled 收益约 +0.09413、分量内容读取收益约 +0.09353，而报告内容收益约 −0.000067。[已有结果记录](igstgnn_f2_formal_results_20261015.md)提供新作用位置的研究动机；不证明入口位置是报告弱效应的原因，也不证明删掉 F2 有益。

## 📋 数据事实与缺失信息

本轮直接读取原发布包的小型文件、邻接矩阵和对象数组头，没有反序列化交通对象、读取真实 X/Y、加载检查点或访问服务器。[实际输入审计](verification/igstgnn_conditional_propagation_inputs_20261015.json)与[审计脚本](../experiments/chronological/audit_min_conditional_propagation_inputs.py)可复核。

| 信息 | 本轮证据级别 | 具体合同 |
|---|---|---|
| 站点轴 | 实际元数据 | 496 个唯一 station_id，与候选图元数据顺序一致；不据此独立认证每个交通张量轴 |
| 对象数组 | 实际 NPY 头 | 5585 条；未读取内部字典；原比例取整为 train 3909 / val 837 / test 839 |
| X/Y 形状 | 已有加载器合同 | X 为 Float32[B,12,496,3]；批处理 Y 仅流量通道；本轮未逐样本重验 |
| 三个交通通道 | 已有代码合同 | flow / nominal tod / nominal dow；不包含速度、密度、库存 |
| 标准化 | 实际 stats 与已有代码 | 使用发布 mean/std，不重新拟合；不新增物理单位认证 |
| 报告类别 | 实际 mapping 与代码 | type 6 类、description 29 类；独立 UNK 编号 6/29，类别 0 有效 |
| 报告关系 | 已有代码合同 | 有限 Float32[B,N,3]；本轮不认证三列实际生成公式或方向物理含义 |
| 时钟 | F2 协议 | clock_offset_slots=0；不得套用旧 chronological 的 +2 |
| 实时可用性 | 未认证 | 继续 published_archive_attributes_conditional_offline |

报告字段严格沿用 `type, description, type_valid, description_valid, relation, relation_available, report_present`。定义

```text
available[b] = report_present[b] AND relation_available[b]
M[b,i] = available[b] AND (sum_c abs(relation[b,i,c]) > 0)
```

缺报告、不可用关系、零支持是三个不同元数据情形，但本候选的事故计算均关闭。有效报告中缺类别才使用注册 UNK。未知类别不是自然无事故；关系零行不是已认证无影响位置。继续排除 `Incident Time, durations, Holiday, incident_position`，不构造事故年龄、解除或最终持续时长。

尚未获得：逐样本绝对日历/跨 split 重叠独立认证、真实报告首次发布字段可用性、关系生成程序、真实容量/队列/密度、原 IGSTGNN 同口径完整预测归档。它们不阻止设计一个条件离线预测算子，但阻止相应的线上、因果和物理主张。

源码证据：[发布加载器](../src/utils/igstgnn_published.py)、[事件编码](../src/utils/igstgnn_published_events.py)、[当前支持规则](../src/models/incident_spatial_forecast_fusion.py)、[冻结协议](../experiments/chronological/min_component_fusion_v1.json)。

## 🧩 模块与张量合同

```mermaid
flowchart LR
    accTitle: Conditional Message Propagation Contract
    accDescr: 一个共享报告编码同时条件化DIF内部消息选择与内容；DIF残差进入INH，两类预测特征经过原读出。此图描述未集成的候选合同。
    history[交通历史] --> encoding[历史状态编码]
    report[报告关系与类别] --> condition[共享报告编码]
    encoding --> condition
    encoding --> diffusion[条件消息传播 DIF]
    condition --> diffusion
    graph[固定图支持与历史图权重] --> diffusion
    diffusion -->|残差历史| temporal[残差时间 INH]
    diffusion -->|forecast| aggregation[预测特征汇总]
    temporal -->|forecast| aggregation
    aggregation --> head[原预测头]
    head --> forecast[十二槽预测]
```

图内 DIF→残差 INH 在五个分解层重复；不是两个独立原始输入分支。每层的 DIF 内部用同一算子完成历史编码与后续三次潜块递推。报告编码在完整 forward 中执行一次，所有分解层及递推调用复用 R。

每层保留原报告无关的 estimation gate：先由残差历史及节点/时钟信息得到 gated history，新核读取该 gated history；backcast 仍从未经过该 gate 的残差历史末Q槽扣除。下文 forecast 窗口的 h 指这一层 gated history。`DIF_hidden`公式给出原dropout之前的输出，原dropout概率0.1保留；条件界与精确比较在eval模式下陈述。

| 接口 | 形状 / 行为 |
|---|---|
| 当前历史表示 H | [B,T,N,32] |
| 一次 valid 窗口 | [B,Q,N,3,32]，Q=T−2 |
| 报告条件 R | [B,N,32]；节点局部计算，不全网池化 |
| 各图/阶权重 | 概念 [B,Q,E_s,3]；实际稀疏边、分块计算 |
| 每个 support bank 消息 | [B,Q,N,32] |
| bank 拼接与投影 | 11×32 → 32；沿用该结构轴，非旧算子函数等价 |
| DIF/INH forecast | 每层各 [B,4,N,256] |
| 统一读出 | 原 gap=3 / 三通道选择 → [B,12,N,1] |

报告编码复用现有 `EventSemanticEncoder` 的 type 4 维、description 8 维、两个 validity flags，投影至 16 维。节点条件输入为当前未受报告影响的编码历史 summary（末槽、均值、末减首，共96维）、三列关系、16维语义和一个 M，共116维；`Linear(116,32)→SiLU→Linear(32,32)→tanh`，最后乘 M。所有带 bias 的运算之后再次硬门控。HISTORY 分组不读取报告，R 全零。不同分解层保留各自核参数，层内历史/forecast 复用该层核。

这里的 tokenwise value 仅保证新核入口处不再先混其他槽；上游 H 本来就可能包含图、INH attention、残差计算形成的上下文。不能把某个 H token 等同一条未经混合的原始观测。

## 🔗 固定图支持与能力取舍

实际发布 `adj_matrix.npy` 为 Float64[496,496]，非负、非对称、对角零，正元素13339。SHA256 与冻结协议一致：`de2a74cc981853ff81dd494d6e7d212b7ae524fb9d6079c91323e56bcb22254f`。

令 S=(A>0)，固定 master 支持为

```text
A_hard = I OR S OR S.T OR bool(S @ S) OR bool(S.T @ S.T)
```

布尔可达计算使用 int32，避免 uint8 计数溢出。它不等于 `(S OR S.T)^2`，后者额外包含混合方向二阶路径。本支持有34208个位置（包含496个 self），密度13.9048%；全部496站位于同一个弱连通分量。因此真实图的“分量隔离”检验不够有区分力，另需人工断连样例。

完整构造链已核对：`IGSTGNN.__init__` 先 `setdefault('use_pre', True)`，再建层。不能仅依据 `STLocalizedConv.get('use_pre', False)` 推断默认关闭。当前实际 bank 结构是 predefined 4 + dynamic 4 + static 2 + self 1 = **11**。这是源码合同核对，未实例化或回放正式模型。

| bank | 硬支持（对角另归 self） | 历史权重来源 |
|---|---|---|
| predefined，方向各一/二阶 | S^k 或 (S.T)^k，k=1,2 | 原 doubletransition 相应图阶 |
| dynamic，方向各一/二阶 | 同上 | 原报告无关历史图对应图阶，随后施加硬支持 |
| static，一/二阶 | S^k OR (S.T)^k | 原节点嵌入静态图对应阶，随后施加硬支持 |
| self | I，三个窗口位置 | 本地值；行尺度1 |

原 learned static 图通常稠密，dynamic mask 是 `adjs+1e-7`；改成硬支持是独立的结构改变，不只是数值修补。它保留双向一/二阶候选与多 bank，牺牲部分旧远程依赖。raw A 的方向只用于预测依赖，不认证真实上下游。

图权重在未受报告影响的完整历史上构造一次，所有层和forecast调用复用；不在事故条件残差上重新生成图。图权重经硬支持后为0的边不进 softmax，不以 epsilon 重新开边。某 bank 的空行消息定义为0；self bank 保证整个算子有本地输入。

严格局部性只陈述为：**固定同一 H、同一历史图、固定参数、eval 模式下，若报告变化导致的 delta R、delta M 仅支撑在连通分量C内，则C外输出不变**。实际type/description是样本级类别；若M覆盖多个分量，更换全局类别会同时改变这些分量的R，不属于单分量干预。原动态图读取全历史，训练 BatchNorm 存在跨节点统计；本合同不承诺一般历史扰动隔离、训练过程隔离或物理传播速度界。参数学习也可以跨全训练集共享信息。

## 🔬 权重与内容的具体公式

固定一个 batch、窗口 q、接收节点 i、support bank s。允许来源 a=(j,ℓ)。令 `A_s(i,j)` 是报告无关的非负硬掩码图权重。图 bank 的 `rho_s=sum_(j,ℓ) A_s(i,j)` 保留其有限行质量；self bank 的 rho=1。空行走0分支。

```text
p_jℓ = [固定窗口位置向量；HISTORY/GENERATED 类型嵌入；生成块编号嵌入]
v_jℓ = W_ℓ H_source(j,ℓ) + b_ℓ                 # 不读取其他槽，允许带符号变换
u_i = tanh(W_q H_receiver(i))
z_jℓ = tanh(W_k H_source(j,ℓ) + W_p p_jℓ)
t_jℓ = tanh(W_t p_jℓ)
r_ij = tanh(W_r [R_i,R_j] + b_r)
gamma_ij = report_enabled AND available AND (M_i OR M_j)

base_score_jℓ = <u_i,z_jℓ>/sqrt(32)
interaction_jℓ = u_i * z_jℓ * t_jℓ * r_ij      # channelwise product
e_jℓ = gamma_ij * beta * tanh(w_e^T interaction_jℓ + b_e)
deltav_jℓ = gamma_ij * eta * tanh(B_v [interaction_jℓ,r_ij] + b_v)

K_H = masked_softmax(log(A_s) + base_score, over allowed j×ℓ)
K_E = masked_softmax(log(A_s) + base_score + e, over allowed j×ℓ)
m_s = rho_s * sum_(j,ℓ) K_E(j,ℓ) * (v_jℓ + deltav_jℓ)
DIF_hidden = W_out [m_self || m_1 || ... || m_10] + b_out
```

所有共同投影的核宽度固定32，沿用 hidden 宽度，不增加 width sweep。`beta=1, eta=1` 是本合同的有限幅度默认值，单位为当前隐表示，不是容量；不通过验证集搜索选值。代数脚本中的其他 beta/eta 仅用于压力样例。事件最终 logit/value readout 零初始化，内部交互因子非零初始化；因此候选自身阶段0 EVENT/HISTORY 可一致，报告路径后续可学习。完整参数量与梯度检查在模型实现时记录，不能引用 MIN 阶段0当作新模型阶段0。

同层的 q/z/t/r、三个位置 value 映射和事件 readout 在11个 bank 共享；各 bank 持自己的图/阶 prior、硬支持和 rho，投影按 bank 分块。分解层之间不共享核参数。时间 p 为11维：三维固定窗口 one-hot、四维来源类型嵌入、四维块号嵌入；历史块号映射到0，G0…G3映射到1…4。这些宽度在本合同固定，不开展选择搜索。

`deltav` 保留 r_ij 的直接输入，所以即使 base values 全相同或全零，也具有产生报告响应的表达可能。它仍在 DIF 的加权消息内，不是额外预测头。gamma 在最终含 bias 映射之后相乘，避免全零输入经 bias 重新产生报告作用。

本合同保证一种非可分离交互的表达可能，不保证训练后一定使用它。位置特定 affine value 与多 bank 拼接保留部分重要能力；旧 `ReLU(W flatten(window))` 跨位置非线性被删除。因此不宣称新函数类包含旧卷积、不承诺旧权重精确暖启动。`W_out` 等形状兼容张量可尝试复用，须逐键记录兼容性。

## ⏱️ 时间轴与截止点合同

| 分解层 d（0起） | 输入T | 输出Q | 原历史槽对齐锚点 |
|---:|---:|---:|---|
| 0 | 12 | 10 | 2…11 |
| 1 | 10 | 8 | 4…11 |
| 2 | 8 | 6 | 6…11 |
| 3 | 6 | 4 | 8…11 |
| 4 | 4 | 2 | 10…11 |

DIF backcast 对齐末Q槽，INH 保持Q。层内窗口q接收位置的原历史锚点为 `2*d+q+2`。锚点不等于该表示只依赖此前前缀；图构造与 INH 已使用完整可观测历史。

每层四个 DIF 潜块严格为

```text
Z_hist = kernel(H)               # 历史窗口已经过新DIF传播算子，仍保留Q轴
z0 = Z_hist[:, -1]               # 不是直接取原输入H[-1]
z1 = kernel([h[-2], h[-1], z0])
z2 = kernel([h[-1], z0, z1])
z3 = kernel([z0, z1, z2])
```

元数据：`kind=HISTORY/GENERATED`；`local_position∈{-2,-1,0}`；`block_id=-1`表示历史，生成块为0…3；历史原槽锚点另外传递供断言。类型与块号使用嵌入，窗口位置使用固定向量。`z0`来自历史末表示，但一旦参与生成递推即标 GENERATED，指用途与来源类型，不赋予 t+1 的物理状态身份。三个递推窗口的 HISTORY 数依次为2/1/0，不允许读取尚未生成的块。

最终只在读出层有确定的目标槽映射：`y_(3b+c)=head_channel_c(sum_d DIF[d,b]+sum_d INH[d,b])`，b=0…3，c=0…2。不是十二次状态递推；不得把 block_id×3 当已认证的真实分钟位置。

R 只依赖预测截止点已有交通和批准的归档报告字段。它可以条件化所有历史表示与生成前缀；不意味着事故提前作用于过去。当前可保证的计算目标是截止点之后不读目标真值；不能宣称每个历史q都满足其自身端点的前缀因果性，也不能宣称归档字段的现实首次发布时间已认证。DIF/INH 只是空间交互/残差时间分工，不是可辨识的物理扩散—本征分解。

## 📐 可证明性质与不能外推的结论

以下为给定公式的条件推导；[代数检查](verification/igstgnn_conditional_propagation_algebra_20261015.json)只给出有限数值见证，不代替证明或完整模型测试。注意力接收某条件却不具有相应选择能力已有理论先例[^1]；输入依赖注意力的稳定性不能由归一化直接推出[^2]。

1. **精确回退。** gamma=0 时 e=deltav=0，当前算子等于同参数HISTORY算子。若编码、图、INH、head无其他报告入口，则固定参数eval下可沿层和递推归纳至完整预测一致。不是独立训练HISTORY权重，也不是旧MIN。
2. **联合关联可改变。** 对一个全允许2×2子表，`D=e_jℓ−e_jℓ'−e_j'ℓ+e_j'ℓ'`。EVENT与HISTORY的邻站×位置 log交叉比之差恰为D，softmax分母与共同graph prior相消。上述乘积存在D≠0的参数；仅可分离事件评分的D恒为0。
3. **同质值盲区被解除。** 对纯logit未缩放均值 `m_bar=sum K*v`，有 `dm_bar/de_a=K_a(v_a−m_bar)`；本合同的m_s再乘报告无关rho，导数也乘rho，空行单列为0。同质v时仍为0。当前内容增量可随r变化，存在零历史下m仍变化的参数；不是每个报告必然改变输出的保证。
4. **单次固定输入扰动界。** 若 `max_a ||v_a||∞≤V`、`|e_a|≤beta`、`||deltav_a||∞≤eta`，则保守地 `||m_E−m_H||∞≤rho*(2*V*min(1,beta)+eta)`。证明：softmax沿线积分的l1变化≤2beta，且两个概率分布l1差≤2；再应用三角不等式。多bank投影后的界可按各W_s诱导范数求和。此处固定同一H、graph、值、参数，且不含随机dropout；不是全网络收缩或报告输入的全局Lipschitz界。
5. **节点重排等变。** 共同重排H、R、图、mask和节点嵌入，使用共享局部函数、同一时间元数据时，该算子输出对应重排。不是不重排节点参数时也成立。
6. **范围界。** 给定报告无关固定图、节点局部R与归一化、无其他报告通道，报告效应只能沿展开计算图允许的边传播。多图阶、五层、INH残差和每层四潜块须一起展开；不写成“五层等于五跳”。

行归一化不保证网络总量守恒，注意力权重不认证真实事故因果、通行容量或时延；残差相减不保证唯一分解。以上性质也不证明超过原IGSTGNN，不能用“理论认证通过”作为模型状态。

## 🧪 实现前后验收与比较

本轮先完成实际小型输入审计和独立NumPy代数检查；没有模型工厂、正式模型调用、优化器更新或测试集评价。

| 必须检查 | 本轮能做的证据 | 完整模型实现后仍需做 |
|---|---|---|
| OFF / unavailable / zero support | 公式级精确回退 | 全五层和四块递推回退、RNG/模式检查 |
| 同质/零值报告响应 | 显式非零参数见证 | 集成模型消息/预测/梯度实际响应 |
| 固定支持的报告内容变化 | 同一M、同一kernel，不同R改变消息 | 真实关系/类别干预与M_ONLY训练对照 |
| 邻站×位置交互 | 2×2交叉比恒等式 | 训练后D分布及是否塌缩 |
| 幅度与空行 | 随机固定输入界、empty row有限 | Float32/GPU/AMP与完整递推压力检查 |
| 图范围 | 严格零边与人工断连见证 | 全展开依赖范围、图分数报告无关检查 |
| 时间和目标读取 | 原合同/prefix静态核对 | 随机/NaN label不变性、原head通道映射 |
| 资源 | score元素算术 | 峰值显存/时间/参数量实际测量 |

资源计算不得忽略channel维：B64、首层Q10、N496、k_t3，稠密union score为472350720个元素；硬支持union score为65679360个元素，Float32约262.7MB，仅一个共享score张量。11 banks按计划支持分别存权重时，共228224个bank边位置，首层438190080个score元素、Float32约1.753GB；这仍不含梯度、交互中间量和优化器。稀疏支持不自动等于低显存或更快。禁止直接物化 `[B,Q,E,3,32]` 的全边消息；必须边分块与融合归约，必要时对训练中间量做checkpoint。只分块却让autograd保留全部中间量，也不能声称解决显存问题。

后续最小研究组：同结构/同初始化规则/同完成预算的 HISTORY、M_ONLY、完整EVENT；M_ONLY保留原M，不从已清零关系重新推断支持。报告OFF只做同检查点依赖；kernel-logit-only / kernel-value-only分别关断时保留同检查点其他计算。F2、旧MIN纯微调及原IGSTGNN保留明确身份。若只有HISTORY提升，则是一般传播结构改进；若完整EVENT不优于M_ONLY，不支持关系/语义增量。

硬支持与旧dense依赖的取舍须单独记录；不能在事故对照两组间改变mask。验证837窗口已参与设计和选模，全部阶段仍为探索性。低流量、MAE/RMSE/MAPE、逐时距及固定困难窗共同报告，不以总体均值掩盖损害。

本轮复核修正：独立数据审查初稿依据叶子类 `.get(...False)` 判断use_pre关闭；根节点沿完整构造链核对后更正为True，最终按11 banks设计。三个独立角色分别审查算子、输入和时间合同；不把审查人数当作证据强度。AI工具帮助设计与复核，最终经验结论仍需正式实验。流程方法来源登记在软件说明[^3]，不作为模型有效性依据。

## 🔗 复核入口与参考

```bash
python experiments/chronological/audit_min_conditional_propagation_inputs.py --data-dir ../data/xtraffic/Contra_Costa
python experiments/chronological/verify_min_conditional_propagation_contract.py
```

脚本默认只输出JSON；指定新的 `--output` 才写文件，已存在输出须显式 `--overwrite`。它们不会训练或运行正式模型。

[^1]: Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? ICLR. https://arxiv.org/abs/2105.14491
[^2]: Kim, H., Papamakarios, G., & Mnih, A. (2021). The Lipschitz Constant of Self-Attention. ICML / PMLR 139. https://proceedings.mlr.press/v139/kim21i.html
[^3]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 软件/研究流程说明；不报告本任务效果认证。https://doi.org/10.48550/arXiv.2609.00065
