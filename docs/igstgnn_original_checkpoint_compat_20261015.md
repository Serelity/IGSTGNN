# 原IGSTGNN：PyTorch 2.3断点加载修复

服务器原版基线的`--resume`在读取`last.pt`时报`Unsupported class collections.Counter`。原因是`MultiStepLR.state_dict()`保留`Counter`类型的milestones，而PyTorch 2.3.1的weights-only允许表不包含该类；本地2.8默认允许，之前本地检查未覆盖这个版本差异。

修复只新增加载兼容驱动，并让两个bash启动器调用它。原训练、评估、runtime和source文件相对`ac277ea`逐字节不变，因此旧运行的严格源码身份校验继续适用，不需迁移或重置identity/checkpoint。

始终保留`weights_only=True`。支持Counter的运行时直接使用原加载器；具备公开`safe_globals`时使用其上下文；已知2.3旧版本只在当前进程的受限加载器允许表中临时添加标准库`collections.Counter`。其他GLOBAL仍拒绝，同名异类拒绝，退出或异常均恢复允许表。没有通用pickle回退，没有转换scheduler为dict，也没有修改优化器、RNG、样本顺序或课程状态。

每次驱动执行保存`loader_compat_*.json`，记录策略、驱动/原入口哈希及前后文件观察。观察发生在原runner锁外，不将它写成跨训练全过程的原子证书；原runner仍负责锁、历史、预算、最优选模与恢复校验。

验证：

- 全部66项CPU测试通过，其中10项兼容测试覆盖2.3缺Counter允许表、完整状态保留、未知eval拒绝、异常/嵌套恢复、已有公共允许项保持、重复milestones普通/显式epoch调度及连续训练和拆分续训全状态一致。
- 本机GPU通过旧`ac277ea`工程运行续训至第三轮；原身份保持，共新增2次工程优化更新。没有测试集评价。
- 另加载官方v2.3.1受限读取器源码，复现同一错误后通过精确Counter允许修复；真实工程`last.pt`全状态值及Counter类型一致，源文件不变，未知eval仍拒绝。该读取器运行在本机2.8序列化容器中，不声称服务器2.3环境已实跑。
- 两个bash启动器通过`bash -n`。验证明细见[记录](verification/igstgnn_original_checkpoint_compat_20261015.json)。

更新修复提交后，在服务器仓库根目录重新运行原命令即可：

```bash
bash experiments/chronological/run_original_igstgnn_baseline.sh --run-dir experiments/chronological_runs/contra_original_baseline_s2025 --resume --epochs 100
```

服务器2.3启动时应先看到`strategy: legacy_2_3_Counter_only`，然后恢复原训练目录。此次错误不意味着已完成轮次丢失；无须新建训练目录或删除`last.pt`。仍按最后一个完整保存epoch恢复，未保存的中断轮重跑。

官方源码：[受限读取器](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/torch/_weights_only_unpickler.py)、[MultiStepLR](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/torch/optim/lr_scheduler.py)。
