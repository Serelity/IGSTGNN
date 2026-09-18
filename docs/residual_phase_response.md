# 可学习的事件级残差阶段响应 C''

状态：`conditional_development`。C'' 是 C' 首轮失败后的新候选，不是精度结论，
也不把阶段系数解释为真实事故效应或因果阶段。

## C' 为什么没有检验到原假设

C' 在 seed 2025 的冻结开发验证集上得到全节点 MAE 22.7586，高于 A 的 22.6867；
事件级和自然日聚类 bootstrap 均确认退化。H7-H12 MAE 从 23.6855 上升到 23.7824。

更关键的是，最佳 checkpoint 中 C' 阶段编码器两层权重分别衰减到约 `1e-25` 和
`1e-27`，门控仍为 `sigmoid(-8)=0.000335`。所有事件实际使用近乎相同的固定曲线。
因此 C' 被判为候选失败，但不能据此判定事件级响应假设失败。

## C'' 参数化

历史状态和事件级空间池化与 C' 相同。编码器输出三条平滑基函数的有符号系数：

```text
c = tanh(MLP([q, a/5]))
r(h) = sum_j c[j] * basis_j(h)
w(h) = max(0, g(h) + (g(h) + epsilon) * expm1(r(h)))
```

其中 `epsilon=0.05` 只用于建立远期梯度，不作为额外预测输入。输出层零初始化，因此
`r(h)=0` 且 `w(h)=g(h)`，模型初始预测与 A 严格一致。对 `r(h)` 的初始梯度为
`g(h)+epsilon`，不会随原高斯在 H7-H12 消失。最终截断保证时间权重非负。

三条基函数仍表示即时、持续和延迟响应，但系数只是预测模型控制量。C'' 增加 128 个
参数。阶段参数使用独立优化组，学习率为 0.002、weight decay 为 0；其余参数保持 A/C'
协议的学习率、weight decay 和调度不变。

## 可学习性证据

训练摘要逐轮保存：

- 阶段参数裁剪前梯度 L1；
- 三个事件系数的均值、标准差和 5/50/95 分位数；
- 各时域响应均值与 5/95 分位数；
- 阶段参数 L1/L2/最大绝对值；
- 相对 A 曲线发生有效变化的事件比例。

若完整训练后系数、参数范数或事件曲线仍接近零，应先判为可学习性失败，不解释精度。
若分支确实学习，再按与 A/B/C/C' 相同的事件配对和自然日聚类 bootstrap 评价。

## 服务器运行

先使用已有 Conda 环境做两轮工程检查：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=3 \
python -u experiments/chronological/train.py \
  --data-dir ../data/chronological/Contra_Costa_v8_dev \
  --output-dir experiments/chronological_runs/contra_phase_residual_cuda_check_01 \
  --protocol experiments/chronological/screening_phase_residual_v3.json \
  --variant phase_residual \
  --device cuda:0 \
  --seed 2025 \
  --check
```

工程检查通过后使用新目录并去掉 `--check`，完整运行 100 轮。筛选仍只使用 seed 2025
和开发验证集；未同时优于 A 的全节点 MAE、H7-H12 MAE 且不实质损害关联节点前，
不运行测试集或追加种子。
