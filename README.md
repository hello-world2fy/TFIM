# TFIM

TFIM / complex RBM 训练结果。

新增 [alpha=2、g=1.5、1000 步结果](TFIM_RBM_alpha2_L10_g1p5_1000/README.md)：L=10、W=1，本次补齐 9 个角度，每个角度搜索 100 个 Optuna trial（含剪枝），保留最优 10 个完整 trial，共 90 个新增最终模型。连同原有四个角度的未改动副本，覆盖 0°～90°、间隔 7.5° 的全部 13 个角度。

- [alpha=2 逐角度汇总](TFIM_RBM_alpha2_L10_g1p5_1000/summary.csv)
- [alpha=2 验收报告](TFIM_RBM_alpha2_L10_g1p5_1000/SUMMARY.md)
- [训练配置、环境证据与恢复方式](TFIM_RBM_alpha2_L10_g1p5_1000/logs/PROTOCOL.md)

alpha=2 的最终参数为 `checkpoint_variables_1000.mpack`，完整日志为 `run_resume_1000.json`。历史环境版本的不确定性、日志与最终参数相差一次更新的能量核对结果，均已记录在验收报告中。

已有 alpha=3 结果：L=10、g=1.0 / 1.5，各包含 0°～90°（间隔 7.5°）共 13 个角度，总计 **26 组、260 个保留 trial**，均从原 1000 步续训至 **1500 步**。续训沿用原 trial 的超参数和优化器状态，使用 8 块 GPU，保留原数据库和全部已有 checkpoint。

结果目录沿用首次交付的名称：[TFIM_RBM_alpha3_L10_18groups_20260927](TFIM_RBM_alpha3_L10_18groups_20260927/README.md)，其中现已包含全部 26 组。

- [1500 步逐组汇总](TFIM_RBM_alpha3_L10_18groups_20260927/summary_1500.csv)
- [续训验收报告](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/SUMMARY.md)
- [g=1 能量曲线](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/g1.0_convergence_1500.pdf)
- [g=1.5 能量曲线](TFIM_RBM_alpha3_L10_18groups_20260927/logs/alpha3_extension_1500/g1.5_convergence_1500.pdf)

读取最终参数使用 `checkpoint_variables_1500.mpack`，完整训练日志为 `run_resume_1500.json`。原 Optuna 数据库仍记录 1000 步搜索结果；1500 步结果另存于每个 trial 的 `extension_1500.json`。g=1、75° 的 trial 12 在续训后能量回升，详见验收报告。

原训练代码来源：[rscortessa/basis-dependence-of-RBM-for-TFIM](https://github.com/rscortessa/basis-dependence-of-RBM-for-TFIM)。源码快照保存在结果目录的 `source_snapshot/` 中。
