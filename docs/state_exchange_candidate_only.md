# 三数据集只运行已有最优模型 GW16 STATE_EXCHANGE

用户已暂停刚开始的成对任务，本入口按新的范围仅运行改进模型。
Alameda与Orange各自从头训练STATE_EXCHANGE，Contra Costa读取已经完成的STATE_EXCHANGE冻结测试缓存。
结构、数据、划分、归一化、描述词表适配和固定100轮配方均沿用既有三数据集候选实现。
不启动原版模型训练、显存检查或验证/测试前向；汇总只含改进模型自身的指标和参数。
没有新原版回放分母，不输出相对原版的提升百分比。

## 运行

使用GitHub固定提交的无Git下载命令，将代码放到新的独立目录。
在igstgnn环境执行：

```bash
bash experiments/state_exchange_multidata_candidate/run.sh
```

默认只新增两套正式100轮训练：Alameda(521节点)、Orange(990节点)。
先执行候选相关数据/模型/续跑测试、四训练/四验证窗口3轮工程检查，
再对同一48窗口训练批做两次Adam更新检查显存，最后训练完整预算、验证并冻结BEST后评估测试。
旧原版基线测试不在该入口的默认测试列表中。
训练过程不创建测试loader，工程检查只用验证集；正式测试不参与BEST选择或追加训练。
每个数据集分别训练，没有从Contra Costa直接加载权重迁移到其他节点轴。

结果根默认位于新代码目录的
`experiments/chronological_runs/gw16_candidate_three_datasets_s2025`。
路径保存到新代码目录的`gw16_candidate_paths.env`。
训练目录保留`<数据集>/candidate/STATE_EXCHANGE/last.pt`结构；
评估另存`candidate_val`和`candidate_test`，避免与历史成对评估混用。

中断后重复相同命令即可恢复完整模型、Adam、调度器、RNG及训练顺序。
已完成训练/检查缓存按身份认证后复用，不重复增加研究训练预算。
部分预算是累计目标：

```bash
GW16_ONLY_EPOCHS=20 bash experiments/state_exchange_multidata_candidate/run.sh
GW16_ONLY_EPOCHS=100 bash experiments/state_exchange_multidata_candidate/run.sh
```

只跑单县可设置GW16_ONLY_DATASETS=Alameda或Orange；沿用同一结果根可逐县补齐。

## 保留已有工作

本次所有代码放在新包state_exchange_multidata_candidate；
已有state_exchange_multidata训练器、模型及其来源哈希保持不变。
旧任务文件保留，不能为切换入口删除旧checkpoint或改写identity。
如旧任务的候选已经开始，可在来源校验通过的前提下将GW16_ONLY_RUN指向旧三数据集结果根，
复用原有`<数据集>/candidate`和`_engineering/<数据集>/candidate`。
新检查/评估使用独立名字，旧原版模型即使存在也不会被该入口启动。

默认数据根为
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/data/xtraffic`。
默认Contra缓存为
`/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025_final_test_locked`。
可通过GW16_ONLY_DATA、GW16_ONLY_CONTRA_TEST和GW16_ONLY_RUN明确设置其他实际路径。

## 结果

```bash
source ./gw16_candidate_paths.env
cat "$GW16_ONLY_RUN/Alameda/candidate_test/report.md"
cat "$GW16_ONLY_RUN/Orange/candidate_test/report.md"
cat "$GW16_ONLY_RUN/three_dataset_candidate_test/report.md"
```

主指标沿用原版12个时距分别计算后等权平均；RMSE不是pooled RMSE。
新预测在算指标前统一C布局，并认证缓存写盘前后的完整数值。
表格只含STATE_EXCHANGE的MAE、RMSE、MAPE、参数量；全部时距另存CSV。
这是单seed固定方法在不同数据集上的表现检验；没有多seed显著性或普适优越结论。
原文测试表只可作为外部参考，不能与验证指标混算。
发布窗口重叠及事故信息首次可用时刻的既有解释边界保持不变。

开发机只编写和静态检查；所有Python、服务器测试、模型计算和数值结果均由服务器执行。
