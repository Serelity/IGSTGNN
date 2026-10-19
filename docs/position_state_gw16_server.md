# GW16：位置与交通状态共同决定的区域／局部演化

2026-10-19。开发分支 `research/position-state-gw16`，从 GW15 提交 `1ac6a24144a8f853b05eee08a5d98106f212e439` 分出。**本地只编写和静态审查，未执行 Python、导入、语法编译、单元测试、模型计算或训练。数值检查和精度结论全部等待服务器。**

## 1. 我们已经超过了什么

此前重复使用的开发验证中，认证原 IGSTGNN 缓存 MAE 为16.288489633，GW9 SUM 为13.368020244，描述性下降约17.93%。这个累计差异包含继续训练、读出调整等收益；不能把它全部归给新的图结构。GW9 BRANCH 比同容量 SUM 只改善0.000120867；GW15 有序适配器比排序适配器只改善0.000232347。历史 OWN4 是诊断参考，筛查失败的旧候选结论均保持。

GW16要回答：**在获得同样输入、位置和训练预算时，位置相关的局部差异如何随交通背景改变其对区域后续变化的作用？这种受约束的状态交换能否超过普通直接预测和自由层次递推？** 当前没有GW16结果，不承诺某个提升百分比。

## 2. 实现的主要改动

- 读取认证发布窗口，12→12、496节点、原全局mean/std。固定道路/方向/里程分区；经纬度、里程、道路、方向和缺失标记形成共享位置编码，按station_id明确对齐。没有按验证误差重新划区。
- 历史编码得到H0，初始化C0=PH0、L0=(I−SP)H0。持久局部状态保持PL=0，归一化只作为更新计算视图。末值和平均绝对变化保留独立条件路径。
- 用显式双线性交互生成交换权重：`w=Q tanh(dot(phi(position),psi(origin,current_C,current_L,lead))/sqrt(8))`。固定位置与推进中的交通状态共同控制两个方向；ORIGIN对照读取固定C0/L0，STATE读取Ch/Lh。
- 十二个真实预测步推进同一组交通状态，事故属性和关系作为内部条件／受支持限制的源；无报告时交通状态仍完整推进。
- 取消旧MIRD的独立十二步参考头及20/5/75阶段。预测为最后观测加区域／局部状态增量的共享线性读出。`reference`仅表示最后观测的持续性锚点，没有参考辅助损失。

这些是机制候选，不宣称位置编码、双线性分解、P/Q或多尺度本身首创。输出聚合是统计关系，不是车辆守恒；隐藏分量不等于已识别的事故因果效应。

## 3. 五组、一次完整结构比较

| 组 | 内容 |
|---|---|
| NATIVE_POS | 完整原IGSTGNN主干，流量embedding后加入同样位置编码；明确的增强母骨干 |
| DIRECT | 相同编码及输入的普通层次直接多时距预测 |
| FREE_RECURRENT | 普通动态层次递推、自由内部局部状态，输出端保持相同区域／局部读出关系 |
| ORIGIN_EXCHANGE | 完整受约束推进，交换条件始终读取起点状态 |
| STATE_EXCHANGE | 唯一候选，交换条件读取当前推进状态 |

四个机制组共享历史、事故、位置和起点条件模块的初始化。ORIGIN与STATE完整初始张量相同。DIRECT/FREE只按活跃参数数目确定头／推进器内部宽度，不用验证误差选宽度，不改变共同历史编码器；实际差异和计算成本归档，容量超2%不能通过筛查。

持久状态和编码器宽度固定32，自更新前馈层内部宽度在任何服务器执行前锁定为96。静态设计审查发现，若候选前馈仅32，等参数FREE输入映射会被挤到约4维，形成过弱对照；当前预算使FREE输入内部宽度约32。实际数目由构造后的活跃参数记录确认。这次修订没有使用新实验结果，不是看验证集调参。

NATIVE_POS增加独立unknown类别，Holiday及incident_position固定为常量，不读取它们的记录值；不可用报告逐样本旁路ICSF并将TIID上下文置零。它是统一输入边界下的新训练对手，**不是旧检查点的原样重放**，也不承诺与历史原模型输出一致。

新组各固定100轮、seed2025、batch48、全程联合、BEST包含epoch0、无早停或课程重启。Adam .002、weight_decay1e-5、eps1e-8，40/70/90轮后lr乘0.5，clip5、dropout0.1。目标为原单位节点MAE+0.2区域MAE，区域预测与目标使用同一有效节点子集，覆盖率至少0.8。有效目标finite且>0.001；MAPE目标至少1。线性预测不临时截断，负值比例另报。历史GW原生mask口径不同，不能将新旧绝对数字混成严格同协议比较。

最低开发门槛：STATE相对DIRECT/FREE/ORIGIN各至少0.5% MAE收益，相对NATIVE_POS不退化；LAST相对自身BEST退化≤1%；高值>=314、时距7–12及流量50–200相对同一个固定强对手均不退化。强对手按三组BEST总体MAE最低选择，并列顺序DIRECT→FREE→ORIGIN。真值分组仅评价，不作为输入或路由。

另报告相对最强机制对手是否达到3%提升，这是研究目标报告，不是效果保证或新增调参依据。0.5%门槛只决定机制后续研究，不等于论文证据充分。失败不自动授权加轮次、改阈值或扩张模块。首版只评价验证集，不提供test入口。

## 4. 服务器获取代码与运行

避免与已有GW9/GW16目录混淆，使用带时间戳的新代码目录；不删除、覆盖旧结果。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
export GW16_CODE="$PWD/IGSTGNN-GW16_$(date +%Y%m%d_%H%M%S)"
git clone --single-branch --branch research/position-state-gw16 \
  https://github.com/Serelity/IGSTGNN.git "$GW16_CODE"
cd "$GW16_CODE"
conda activate igstgnn
export GW16_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
export GW16_RUN="$PWD/experiments/chronological_runs/contra_gw16_pos_state_s2025"
git log -1 --oneline
printf 'GW16_CODE=%s\nGW16_DATA=%s\nGW16_RUN=%s\n' "$GW16_CODE" "$GW16_DATA" "$GW16_RUN"
bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
```

默认先运行新包的服务器单测、真实batch工程检查，再运行**五组各100轮**；工程检查失败立即停止。首轮建议按服务器规范在作业或持久会话中运行，耗时以实测为准。

若数据目录不同，用六文件指纹只读定位；不要重建split或重新归一化：

```bash
export GW16_DATA="$(python -m experiments.position_state_gw16.locate_data \
  --search-root /seu_share/home/huangkai/220243809/paper/IGSTGNN)"
```

只做独立工程演练，不进入研究预算：

```bash
ENGINEERING=1 bash experiments/position_state_gw16/run.sh "$GW16_DATA" "${GW16_RUN}_engineering"
```

## 5. 续跑与分段

同代码、数据、配置、设备及运行时，恢复全部优化器/调度器/RNG/样本顺序。从上一完整轮继续，不重置优化器：

```bash
cd "$GW16_CODE"
RESUME=1 bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
```

如果担心作业时限，可先到总计20轮，再继续到总计100轮。不是增加100轮：

```bash
EPOCHS=20 bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
RESUME=1 EPOCHS=100 bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
```

部分预算的评价存入带epoch后缀的独立目录，不能当正式完整筛查。训练和评价不能在同一run并发；锁会拒绝。初始化中断且缺identity/runtime时保留现场，换新run，不伪造续跑身份。

## 6. 看结果

完整运行后：

```bash
cat "$GW16_RUN/summary.json"
cat "$GW16_RUN/eval_best_val/report.md"
cat "$GW16_RUN/eval_last_val/report.md"
cat "$GW16_RUN/screening.json"
```

单独重建评价（同检查点的已认证缓存会复用，不重新推理）：

```bash
python -m experiments.position_state_gw16.evaluate --data-dir "$GW16_DATA" \
  --run-dir "$GW16_RUN" --selection best
python -m experiments.position_state_gw16.evaluate --data-dir "$GW16_DATA" \
  --run-dir "$GW16_RUN" --selection last
python -m experiments.position_state_gw16.screen --run-dir "$GW16_RUN"
```

`record.log`及各组`history.json`包含epoch0和训练记录；身份、位置归一化、分区、源码指纹及BEST/LAST预测归档随run保存。评价和筛查校验权重、同轴标签、身份及缓存指纹。开发验证反复用于设计，完整实验仍属于开发证据；后续独立确认需要方法冻结，不把重叠窗口当独立事故样本。
