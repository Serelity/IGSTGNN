# 原版 IGSTGNN：双卡并行训练 Alameda 与 Orange

本入口按用户确认的范围，只训练原版 IGSTGNN 的 Alameda 和 Orange。
每个数据集使用独立进程、GPU、日志和检查点；没有改进模型训练，也不依赖 Contra Costa 的权重或结果。
原版模型来自已冻结的作者源码，复用现有跨数据集原版训练器，未改动其源码指纹。

## 启动

在服务器已分配两张 GPU 的终端中，进入本分支代码目录，激活 igstgnn 环境后运行：

```bash
conda activate igstgnn
bash experiments/original_multidata_only/run.sh
```

默认 Alameda 使用 cuda:0，Orange 使用 cuda:1；编号相对于当前 CUDA_VISIBLE_DEVICES。
脚本保留调度器提供的可见设备设置，不替换成物理卡编号。
需要调整时，在启动命令前设置 IGSTGNN_ORIGINAL_ALAMEDA_DEVICE 和 IGSTGNN_ORIGINAL_ORANGE_DEVICE。
它们必须是两张不同且当前可见的 GPU。

数据根默认：
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic`。
可通过 IGSTGNN_ORIGINAL_DATA 指定其他数据根；六个发布文件的内容仍须通过已登记的 SHA256 验证。
本地 Git 仓库不包含服务器数据。

结果根默认位于当前代码目录：
`experiments/chronological_runs/original_alameda_orange_s2025`。
可用 IGSTGNN_ORIGINAL_RUN 指定固定的其他结果根。
两个 worker 的日志、正式训练和评估产物相互独立；其中一个失败时另一个继续运行。

## 训练与结果口径

原版训练器沿用项目既有复现配方：seed 2025、batch 48、Adam、原版课程学习、学习率调度、
最多 100 轮及 patience 20 的早停。这里的 batch 48 沿用项目复现配置，不能描述成作者 CLI 全部默认值。
不加载 Contra Costa 权重迁移，不向原模型添加位置或状态交换模块。
验证集按原版 pooled MAE 选择 BEST；不把未训练的 stage zero 纳入原版 BEST。

入口先运行原版相关单元测试，再让两个数据集各自在其 GPU 上完成小规模工程检查、
同一 48 窗口训练批的两次 Adam 更新显存检查和正式训练。
工程检查只有四个训练、四个验证窗口及三轮训练，不评估测试集。
完整 batch 检查失败时保留诊断，不自动缩小 batch 或更改训练配方。

正式训练满足原版完成条件后，分别回放验证集并评估测试集。
测试前冻结验证选出的 BEST、来源文件和数值环境；结果目录采用认证缓存。
训练与 BEST 选择不使用测试指标。重复运行只补齐缺失结果，不增加已经完成的训练预算。

主表为每个时距分别计算原版 FP32 MAE、RMSE、MAPE 后，对 12 个时距等权平均。
RMSE 不是把全部时距混在一起计算的 pooled RMSE。
各时距结果写入 per_horizon.csv；补充 pooled 口径另存 JSON。
两个数据集的单独原版报告不计算相对改进模型的提升率。

## 查看和续跑

路径记录在代码目录的 original_multidata_paths.env。查看结果：

```bash
source ./original_multidata_paths.env
cat "$IGSTGNN_ORIGINAL_RUN/Alameda/original_test/report.md"
cat "$IGSTGNN_ORIGINAL_RUN/Orange/original_test/report.md"
```

中断后，保持代码、设备编号、线程数和结果根不变，再运行同一条 bash 命令即可续跑。
如使用了自定义结果根，重新设置并导出该变量后再启动：

```bash
export IGSTGNN_ORIGINAL_RUN="/实际已有的结果根"
bash experiments/original_multidata_only/run.sh
```

续跑恢复模型、Adam、调度器、课程学习进度、随机状态和样本顺序。
检查点按完整 epoch 原子提交；中断发生在 epoch 内时，重做该 epoch。
如果初始化在 identity 提交前就中断，脚本报出该目录并停止，保留现场，不能把缺少身份的目录当成可续跑结果。
正常 Ctrl+C 会停止本入口启动的两个 worker；不会结束其他实验进程。

本次开发机仅进行静态检查与 Bash 编排检查，不执行 Python 或模型。
数值单元测试、实际显存与完整训练结果均由服务器入口执行后产生。
原版和改进模型即使具有同一发布样本轴，也仍需分别认证最终结果后再做正式对照。
