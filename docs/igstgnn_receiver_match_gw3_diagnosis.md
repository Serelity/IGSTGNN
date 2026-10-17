# GW3 下一步：只读损害定位

本轮实现的是模型改动前的诊断工具。没有新增模型版本、损失项或参数搜索；不重训、不新增推理、不读原始数据或检查点、不评价测试集。代码与测试均未在本地运行，服务器验证待完成。

## 为什么先诊断

用户回传的五轮结果：RECEIVER_MATCH MAE 15.340547214，优于MESSAGE_SELF的15.912978592；关闭动态匹配时变为15.493144392。但GW2 POST_MESSAGE仍更好，为15.156222029。FIXED_UNIFORM最佳第4轮16.352451620，第5轮变为17.260126036。

因此，需要区分接收查询的同检查点依赖、联合训练模型的整体效果和固定组末轮波动，不能只引用最有利的一组差值继续堆模块。此工具为选择下一处结构改动补齐已有证据。

## 服务器更新与运行

在原GW3目录更新同一分支；不需要切换GW1/GW2目录，也不需要GPU。保留原运行及其GW2/GW1来源目录。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW3
git fetch origin
git merge --ff-only origin/research/receiver-match-gw3
conda activate igstgnn

export GW3_RUN=experiments/chronological_runs/contra_original_gw3_s2025
export GW3_DIAG="$GW3_RUN/read_only_diagnosis_$(date +%Y%m%d_%H%M%S)"

python -m unittest tests.test_receiver_match_diagnosis -v && \
python -u experiments/chronological/analyze_receiver_match_gw3.py \
  --run-dir "$GW3_RUN" \
  --output-dir "$GW3_DIAG"
```

自动寻找与本次训练身份、完成预算、best/last角色相符的已有评估。缺少认证归档直接报错，不补跑评估。若有多个目录并需固定来源，可在诊断命令加 `--best-eval "$GW3_BEST_EVAL" --last-eval "$GW3_LAST_EVAL"`，两个变量先设置为已有评估实际目录。

通过后读取：

```bash
cat "$GW3_DIAG/summary.json"
cat "$GW3_DIAG/report.md"
cat "$GW3_DIAG/mechanism_availability.json"
```

正常状态为 `GW3_READ_ONLY_DIAGNOSIS_PASS`，新增推理和优化器更新均为0。即使部分机制统计缺失，已认证预测的误差分析仍可通过；缺失项及原因单独列出，不用0冒充已观测值。

## 输出回答哪些问题

| 制品 | 内容 |
|---|---|
| report.md | 全局收益、主要损害组、逐轮曲线和解释限制 |
| best_comparisons.csv / last_comparisons.csv | 模型间比较、关闭选择器与查询替换依赖、正负抵消 |
| best_groups.csv / last_groups.csv | 流量、道路、报告支持、GW2高低估、固定困难集 |
| best_horizons.csv / last_horizons.csv | 每个比较的12个实际预测时距 |
| best_group_horizons.csv / last_group_horizons.csv | GW3相对GW2的分组×实际预测时距 |
| best_road_flow.csv / last_road_flow.csv | GW3相对GW2的道路×真实流量 |
| best_windows.csv / last_windows.csv | 每窗口收益及统一困难集标记，不把窗口当独立事故 |
| best_directions.csv / last_directions.csv | 朝向目标、远离目标、过冲等逐点误差方向 |
| best_error_identity.csv / last_error_identity.csv | 各组同点绝对误差恒等式核对 |
| training_curves.csv / matched_epoch_curves.csv | 原有六组完整逐轮记录、同轮次GW3相对GW2差值 |
| training_mechanisms.csv | 各轮已保存的按调用阶段跨样本变化 |
| mechanism_summary.csv / mechanism_phase.csv | 选中检查点的总体统计与固定节点/窗口的跨样本统计 |
| mechanism_availability.json | 缓存机制来源链与缺失原因 |
| preservation.json | 所有已读文件的SHA256与运行后未变更核对 |

同检查点的比较复用同一真值、原mask、样本/节点轴。best和last的轴、真值、道路及支持必须完全一致。困难集只按best ORIGINAL最差10%完整窗口选定，样本ID升序打破同分，所有模型及last都复用这些ID。

流量、道路、GW2高低估各自作为完备分区，分别核对全网贡献守恒。高低估分组固定由对应选择状态的GW2 POST_MESSAGE与真值定义，各比较不自行更换该分组。全网贡献分母始终是全网所有时距有效点数；同组不同实际时距的贡献可相加。不同分组体系不能混加。

正负抵消按逐点绝对误差收益计算；保留负消融与空组。MAPE在JSON/CSV中为比例，报告换算为百分数/百分点。0.001仅为预测响应诊断阈值，不用于在线决策，也不宣称已认证物理单位。

## 来源保护与解释边界

工具只导入NumPy及标准库，不导入训练器、模型或torch。实际只读JSON、NPZ与保存生产者的源码哈希，拒绝读取.pt；不调用torch.load或任何模型接口。检查归档SHA256、训练身份、已有回放声明、检查点角色、完成预算、缓存基线预测相等，以及保存历史与当前指标的一致性。这是归档内部一致性核对，不是新检查点回放或独立历史签名认证。

机制统计可能因best/last缓存复用只保存指针。工具在已声明GW3/GW2运行目录内递归追溯机制JSON，拒绝循环和越界来源。总体weight_std混合多个轴；优先查看按history、forecast_1等阶段固定节点/窗口计算的across_sample_std_rms。调用阶段不等于真实预测时距，不能直接把forecast_1的统计解释为第1步预测原因。

归档没有逐点权重极值、查询/键范数、评分或梯度，所以不能判定局部饱和、尺度爆炸、梯度病因，也不能将平均熵解释为可靠性。工具不会为补齐这些量新增推理。

逐点恒等式：完整模型误差收益=固定汇聚干预误差收益+GW3选择器依赖−GW2选择器依赖。仅对绝对误差及其MAE汇总成立；不是模块训练因果分解，不适用于RMSE，也不证明主干被损坏。

后续只在诊断结果支持时选择一个模型结构假设；本工具不自动训练、选模型、调参或以开发验证集宣称独立测试提升。
