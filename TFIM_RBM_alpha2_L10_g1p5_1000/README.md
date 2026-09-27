# TFIM complex RBM：α=2，L=10，W=1，g=1.5

新增9个角度已完成：0°、15°、22.5°、37.5°、45°、60°、67.5°、82.5°、90°。每个角度100个Optuna trial（含剪枝），保留最优10个完整trial，各1000步。没有加训到1500步。

本交付文件夹同时收录原有7.5°、30°、52.5°、75°的未改动副本，共覆盖0°～90°、间隔7.5°的全部13个角度。

- [验收汇总](SUMMARY.md)
- [逐角度表格](summary.csv)
- [运行协议及证据限制](logs/PROTOCOL.md)
- [逐trial能量核对](logs/trial_energy_validation.csv)
- [13角度训练曲线](logs/convergence_13_angles.pdf)

`PLAYING_WITH_RBM_COMPLEX/` 保持原命名层级、数据库和 checkpoint 格式。`optuna_trials` 为入选10个trial；`_preserved_trials` 和 `_recovery` 是额外保留的归档/恢复数据，不计入入选数量。

`summary.csv` / `summary.json` 中路径相对于本文件夹；原日志中的绝对路径为来源记录。`logs/` 保存完整训练日志、配置、环境版本、源码快照及校验记录，`scripts/` 是任务调度/恢复/验收脚本（保留原工作区路径，迁移运行前需调整）。

历史NetKet/JAX/Python版本未锁定，不能宣称环境逐版本一致或逐位复现；实际采用原源码训练/搜索流程、相同预算与精度。原流程未固定训练随机种子。末条日志与最终参数相差一次更新，能量差异已单独记录，原始数据未改写。

可在本目录运行 `sha256sum -c SHA256SUMS` 验证完整性。
