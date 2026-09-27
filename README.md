# TFIM

TFIM / complex RBM 的 alpha=3 补训结果。

本次包含 L=10、g=1.0 / 1.5 的 18 组实验；每组执行 100 个 Optuna trial（含剪枝），保留 10 个训练至 1000 步的 trial。

完整结果位于 [TFIM_RBM_alpha3_L10_18groups_20260927](TFIM_RBM_alpha3_L10_18groups_20260927/README.md)，包括数据库、中间及最终 checkpoint、汇总、验证图、日志和脚本。

- [结果汇总](TFIM_RBM_alpha3_L10_18groups_20260927/summary.csv)
- [训练流程](TFIM_RBM_alpha3_L10_18groups_20260927/logs/PROTOCOL.md)
- [验收记录](TFIM_RBM_alpha3_L10_18groups_20260927/logs/validation_result.json)

原训练代码来源：[rscortessa/basis-dependence-of-RBM-for-TFIM](https://github.com/rscortessa/basis-dependence-of-RBM-for-TFIM)。源码快照保存在结果目录的 `source_snapshot/` 中。
