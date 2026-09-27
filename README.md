# TFIM

TFIM / complex RBM 的 alpha=3 训练结果。

最新结果：L=10、g=1.0 / 1.5，各包含 0°～90°（间隔 7.5°）共 13 个角度，总计 **26 组、260 个保留 trial**，均从原 1000 步续训至 **1500 步**。续训沿用原 trial 的超参数和优化器状态，使用 8 块 GPU，保留原数据库和全部已有 checkpoint。

结果目录沿用首次交付的名称：[TFIM_RBM_alpha3_L10_18groups_20260927](TFIM_RBM_alpha3_L10_18groups_20260927/README.md)，其中现已包含全部 26 组。

- [1500 步逐组汇总](TFIM_RBM_alpha3_L10_18groups_20260927/summary_1500.csv)
- [续训验收报告](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/SUMMARY.md)
- [g=1 能量曲线](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/g1.0_convergence_1500.pdf)
- [g=1.5 能量曲线](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/g1.5_convergence_1500.pdf)

读取最终参数使用 `checkpoint_variables_1500.mpack`，完整训练日志为 `run_resume_1500.json`。原 Optuna 数据库仍记录 1000 步搜索结果；1500 步结果另存于每个 trial 的 `extension_1500.json`。g=1、75° 的 trial 12 在续训后能量回升，详见验收报告。

原训练代码来源：[rscortessa/basis-dependence-of-RBM-for-TFIM](https://github.com/rscortessa/basis-dependence-of-RBM-for-TFIM)。源码快照保存在结果目录的 `source_snapshot/` 中。
