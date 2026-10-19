# GW16 position-conditioned regional/local state evolution

完整模型、数据边界、预算、续跑和结果查看见 [服务器说明](../../docs/position_state_gw16_server.md)。

```bash
bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
RESUME=1 bash experiments/position_state_gw16/run.sh "$GW16_DATA" "$GW16_RUN"
```

五组从头训练、每组100轮；同位置信息和统一目标。唯一候选STATE_EXCHANGE；DIRECT、FREE_RECURRENT、ORIGIN_EXCHANGE为机制对手，NATIVE_POS为显式位置增强的原主干。固定元数据位置与动态推进状态通过双线性交互控制跨尺度交换。

默认验证集，首版没有test入口。旧结果不再训练、推理或拟合查表。新模型非嵌套于旧冻结源，不声称恢复旧检查点。

本地仅编写与静态审查。没有在开发PC运行Python、导入、编译、单测或模型；服务器检查通过前，不称数值验证完成。
