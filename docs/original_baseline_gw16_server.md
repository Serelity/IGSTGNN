# GW16 主基线纠正：未改结构的原版 IGSTGNN

2026-10-19。本说明纠正原 GW16 交付文档中的主对照安排及历史数值解释。此次只新增独立入口和说明，不修改 `experiments/position_state_gw16/`、其运行依赖、配置、源码身份或已有结果。

## 对比对象

主对比是原论文完整 IGSTGNN 与改进模型。原论文 Contra Costa 测试平均 MAE 13.43、RMSE 22.50、MAPE 15.84%，见 [原文 Table 4/5](https://arxiv.org/html/2602.02528v2)。当前开发的验证结果不能直接与这些测试结果算提升。

`NATIVE_POS` 是补充控制：目的是检查新增显式位置输入可能带来的收益。它除了位置编码，还改变了事故缺失值处理、输入边界及训练配方，不能替代原模型。原版自身已有图、节点嵌入及事故到传感器空间关系；不加 GW16 位置模块不意味着原版完全没有空间信息。

历史 GW 缓存 ORIGINAL=16.288489633 来自 GW1 仅5轮、410次更新、BEST=epoch5，不能代表充分训练的原版或论文结果。此前相对它计算的17.93%累计下降不能称为超过论文的提升。另有旧原版完整训练记录与权重，应优先复用；没有权重时才新训原版。

此次独立入口复用已有 `original_igstgnn_source.py` 锁定的 558ab01 源码和原 parser/model/engine，注册参数455220。原始架构、ICSF/TIID、事故字段、传感器解析、图、原生损失都保持原样；没有添加位置编码或区域损失，也不使用公式对齐版冒充原版。

原版训练最多100轮、patience20；Adam .002、batch48、seed2025，原学习率里程碑 `[1,38,46,54,62,70,80]`，warm30/cl3课程。训练中不评估测试集。GW16 使用自己的100轮无早停联合配方，因此这两者属于完整系统主比较，不能冒称训练配方完全相同或将差异全归给结构。新入口不改变 GW16 已锁定的内部机制比较。

## 当前正在跑的 NATIVE_POS

不必先跑完 NATIVE_POS 才能运行原模型。如果要优先原模型，在当前前台训练终端按一次 `Ctrl+C`，等待进程退出。已提交的 `NATIVE_POS/last.pt`、BEST 和 history 保留；未完成的一轮以后重做。不要删除目录、改名权重、将其重命名为 ORIGINAL，或修改正在运行的源码。

这是补充对照的次序调整，不是原筛查完成。以后仍可使用旧代码目录与原 `GW16_RUN` 正常续跑。跳过 NATIVE_POS 会使原五组100轮 screening 的完整性条件不成立；此次新入口不修改该条件，也不将它设为通过。

## 获取补充代码

保留当前代码目录和 `GW16_RUN`，另开一个带时间戳的代码目录，避免混淆。无需删除或覆盖旧训练。

```bash
cd /seu_share/home/huangkai/220243809/paper/IGSTGNN
export GW16_BASE_CODE="$PWD/IGSTGNN-GW16_BASELINE_$(date +%Y%m%d_%H%M%S)"
git clone --single-branch --branch research/position-state-gw16 \
  https://github.com/Serelity/IGSTGNN.git "$GW16_BASE_CODE"
cd "$GW16_BASE_CODE"
conda activate igstgnn
export GW16_DATA="/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic/Contra_Costa"
```

若数据不在此目录，沿用此前已认证的 `GW16_DATA`，不要重建数据或重新拟合 scaler。

## 优先复用完整原版权重

先只查找原版历史训练，不启动训练或推理：

```bash
find /seu_share/home/huangkai/220243809/paper/IGSTGNN -maxdepth 9 -type f \
  -path '*Contra_Costa_igstgnn_s2025_20260908_193127*' \
  \( -name 'final_model_s2025.pt' -o -name 'record*.log' -o -name '*.zip' \)
```

本地已有的完整历史原版 ZIP 名为 `Contra_Costa_igstgnn_s2025_20260908_193127_20260909090248.zip`；不要求为代码交付上传 ZIP。若服务器已有该 ZIP，可以直接读取它，不解压、不训练：

```bash
export ORIGINAL_ARCHIVE='/实际找到的原版ZIP绝对路径'
export ORIGINAL_EVAL="$PWD/experiments/chronological_runs/original_native_val_$(date +%Y%m%d_%H%M%S)"
python -m unittest tests.test_original_baseline_gw16_evaluate \
  tests.test_original_baseline_gw16_compare -v
python -m experiments.original_baseline_gw16.evaluate \
  --archive "$ORIGINAL_ARCHIVE" --data-dir "$GW16_DATA" \
  --output-dir "$ORIGINAL_EVAL" --device cuda:0 --threads 3
```

若找到的是 `final_model_s2025.pt` 和它原训练的日志，改用：

```bash
python -m experiments.original_baseline_gw16.evaluate \
  --checkpoint '/实际找到的final_model_s2025.pt绝对路径' \
  --training-log '/同一次原版训练日志的绝对路径' \
  --data-dir "$GW16_DATA" --output-dir "$ORIGINAL_EVAL" \
  --device cuda:0 --threads 3
```

入口严格核对原模型类别/结构/权重、训练日志、数据指纹与验证回放。回放失败会停止，不放宽容差。历史日志不是旧服务器精确执行提交的独立证明；历史回放、部分训练和完整新训练的身份在报告中区分。不要把5轮GW1当成完整原版来源。

## 没有可复用的完整权重时才训练原版

原版独立训练目录，不使用 `GW16_RUN`，不从 NATIVE_POS 权重初始化：

```bash
export GW16_ORIGINAL_RUN="$PWD/experiments/chronological_runs/contra_original_native_s2025"
bash experiments/original_baseline_gw16/run.sh "$GW16_DATA" "$GW16_ORIGINAL_RUN"
```

该命令先运行相关服务器单测，再运行现有原版训练入口；最多100轮或原生patience20早停，然后只回放一次 FULL 验证预测，不重跑12组模块消融。中断后保留同一原版目录续跑：

```bash
RESUME=1 bash experiments/original_baseline_gw16/run.sh "$GW16_DATA" "$GW16_ORIGINAL_RUN"
```

如果已经有 `original_igstgnn_baseline_training_v1` 原版run，可以直接回放：

```bash
python -m experiments.original_baseline_gw16.evaluate \
  --source-run "$GW16_ORIGINAL_RUN" --data-dir "$GW16_DATA" \
  --output-dir "$ORIGINAL_EVAL" --device cuda:0 --threads 3
```

读 `report.md`、`metadata.json`。metadata记录训练完成状态、原生mask和回放证据。训练日志依然是原生pooled指标，不能直接与GW16不同mask的日志当作严格配对。

## 与已保存的 GW16 预测统一评分

两种模型均使用同一验证目标、样本和站点轴，分别输出：原生 pooled、原生逐步指标与12步等权平均、GW16共同mask的 pooled。原生逐步指标使用全验证集确定的同一个null值；RMSE先各步开方再平均，不用pooled RMSE替换。

当 GW16 有已认证的某一组预测后，下面比较只读取两份缓存，不再做模型推理。`GW16_ARM_EVAL` 必须指向具体组目录，而不是上层报告目录：

```bash
export GW16_ARM_EVAL="${GW16_RUN:?恢复原GW16_RUN绝对路径}/eval_best_val/STATE_EXCHANGE"
export ORIGINAL_COMPARE="$PWD/experiments/chronological_runs/original_vs_state_val_$(date +%Y%m%d_%H%M%S)"
python -m experiments.original_baseline_gw16.compare \
  --original-eval "$ORIGINAL_EVAL" --gw16-eval "$GW16_ARM_EVAL" \
  --output-dir "$ORIGINAL_COMPARE"
cat "$ORIGINAL_COMPARE/report.md"
```

分段GW16评价目录可能是 `eval_best_val_epoch_XXX/STATE_EXCHANGE`，必须填实际已存在的目录。partial或工程结果可用于观察，但不能当完整100轮比较。比较报告不改变GW16筛查，不因结果不好增加训练预算，也不打开测试集。最终方法冻结后，另行补充统一测试集入口，才与论文13.43正式对标。

## 本地验证范围

此次开发本地不执行 Python、语法编译、单元测试或模型计算。仅作文本、JSON/哈希、Git diff 和 Bash `-n` 静态检查。新增单测与所有数值回放必须在服务器运行；代码交付不等于已经获得新的精度结果。
