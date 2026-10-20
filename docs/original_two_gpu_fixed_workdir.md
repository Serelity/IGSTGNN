# 从 IGSTGNN-code 工作目录提交原版双卡任务

作业页面的工作目录固定为：

```text
/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code
```

在该目录保存 scripts/run_original_two_gpu_from_igstgnn_code.sh 的副本，命名为 run_original_two_gpu.sh。
登录节点先执行一次 bash run_original_two_gpu.sh --prepare-only，准备或核对固定 ea13bc4 发布版代码。
该操作不运行 Python，不训练，也不要求登录节点有 GPU。
下载只在代码不存在时发生，已存在的工作代码和实验文件不覆盖、不切换分支。
准备失败的下载目录保留并打印路径，便于定位网络或文件系统问题。

提交页面设置 igstgnn 环境、两张 GPU，启动命令只填：

```bash
bash run_original_two_gpu.sh
```

脚本自动设置数据、结果、两个设备和线程路径，然后启动已经发布的原版双卡入口。
Alameda 使用当前可见设备 cuda:0，Orange 使用 cuda:1。
沿用上一版结果根，避免切换提交目录后另起重复训练：

```text
/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-ORIGINAL-2GPU_ea13bc4/experiments/chronological_runs/original_alameda_orange_s2025
```

入口路径保存在 IGSTGNN-code/original_multidata_job_paths.env。
重复同一启动命令由原版训练器恢复完整检查点，完成后的结果按原协议复用。
如需沿用自定义结果根，可将对应 export 写在本地 run_original_two_gpu.sh 的开头。
本次只新增外层工作目录入口；冻结的原版代码、双卡训练脚本、评估来源指纹和已有检查点协议均未改动。
