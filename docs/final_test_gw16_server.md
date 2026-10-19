# GW16 冻结 BEST 后的原版配对测试

开发机不运行 Python。以下单测和数值评估全部在既有 `igstgnn` 服务器环境执行。
本次仅新增兄弟包，不修改任何已训练 GW16 源码、配置或原版实现，旧身份哈希继续有效。

## 已完成的训练和验证

用户于2026-10-19提交的完整输出记录训练到100轮，验证 BEST 第99轮。
同样的原生12步等权平均口径：

| 方法 | val MAE | val RMSE | val MAPE % |
|---|---:|---:|---:|
| 原版 IGSTGNN | 13.865727504 | 23.235885779 | 19.506001348 |
| STATE_EXCHANGE | 12.529132684 | 21.583388964 | 17.448496694 |

验证 MAE 改善9.639558%。GW16共同掩码 pooled MAE BEST为12.528962405，LAST为12.53354106。
这些是开发验证结果，不能直接与论文测试13.43比较。单seed整体方法收益尚不能独立归因于某一个结构。

## 入口及固定输入

从本次交付的 GitHub 固定提交下载源码到一个新目录，然后在该目录运行：

```bash
conda activate igstgnn
bash experiments/final_test_gw16/run.sh
```

不需要 `.git`、git clone 或 git pull。交付回复提供精确的 codeload 提交链接与下载指令。
运行脚本默认使用用户已确认的实际路径：

- 旧代码：`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW16_BASELINE_1d0lEe0V`
- 完整训练：旧代码下 `experiments/chronological_runs/contra_gw16_state_main_s2025`
- BEST验证缓存：该训练目录下 `eval_best_val_epoch_100/STATE_EXCHANGE`
- 原版验证缓存：旧代码下 `experiments/chronological_runs/original_native_val_20261019_223356`
- 正式测试：完整训练目录的同级目录，名字追加 `_final_test_locked`

如果移动过目录，可显式设 `GW16_PRIOR_CODE`、`GW16_DATA`，或更具体的
`GW16_FINAL_SOURCE_RUN`、`GW16_FINAL_VAL`、`GW16_FINAL_ORIGINAL_VAL`。
线程数和设备必须匹配候选训练记录，当前默认 `cuda:0`、3线程。
这些变量只定位既有完整成果，不能更换模型、配置或按测试结果选轮数。

## 数据和指标

两模型使用经过六文件SHA256认证的同一作者发布数据，顺序切片 `[4746:5585]`，
全部839个测试窗口、496节点、12步历史预测12步未来，batch48，最后一批23。
复用原有标准化、原始传感器解析器、GW16位置变换和区域划分，不拟合测试集。
原版采用原生日志已验证的历史完整训练权重，不使用 NATIVE_POS。
原版和GW16两种输入表示及训练配方保留，结果是整体方法对比。

主指标使用冻结原版的指标函数，在完整测试目标上确定一次 native null，然后逐步算
MAE/RMSE/MAPE，将12步等权平均。RMSE逐步开方后平均；另存 native pooled 和 GW16 common pooled，
避免混淆。输出全部12步以及第3、6、12步报告，不截断负预测、不后验校准。
论文Contra Costa外部参考13.43 / 22.50 / 15.84%已核对
[表4/5](https://arxiv.org/html/2602.02528v2)，不用于配对改善率的分母。

## 冻结和中断恢复

构造测试loader前，`freeze.json` 固定：验证选中的BEST、原版权重/日志、双方验证缓存、
候选训练身份与所有执行源码、六文件数据指纹、数值runtime、样本轴、指标规则。
先验证完整100轮和原版完整训练证据；没有 test 选模、训练更新或自动追加预算。
测试目标绝不传入两模型forward。推理前后检查注册参数/缓冲区、RNG和原始batch不变。

每模型完整预测先写独立staging目录，再原子发布；已完成的缓存重算指标并核对所有指纹后复用。
中断后，在**同一下载目录、同一环境、同一输出**再次运行同一个 `run.sh` 即可。
一份缓存完成后中断，下一次仅评估另一模型；两份都完整则零模型前向。
模型内部批次尚未完整保存时，只能重做该模型的只读推理，不会训练。
源码、权重、数据、runtime或选模记录发生变化时拒绝覆盖。
残缺已命名模型目录不自动清理；异常 `.writing_*` 与 failure记录保留供工程排查。

## 查看结果

运行脚本保存新代码目录下 `gw16_final_test_paths.env`。同一个新代码目录执行：

```bash
source ./gw16_final_test_paths.env
cat "$GW16_TEST_RUN/report.md"
cat "$GW16_TEST_RUN/comparison.json"
cat "$GW16_TEST_RUN/per_horizon.csv"
```

`GW16_FINAL_TEST_COMPLETE` 才表示双方缓存认证并完成同轴配对。不会重跑验证前向或之前GW实验。
已有历史原版测试数字，当前候选由复用开发验证决定；本次保留测试不等于多seed统计确认。
发布窗口存在历史重叠，不能宣称严格时间/事件独立或事故因果效应。
机制消融需另行处理，未完成其它四组不等于五组筛查通过；本入口不改既有筛查结果。

本次只编写服务器测试并做开发机静态检查；交付前未执行Python或数值测试。
