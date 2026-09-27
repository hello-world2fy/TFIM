# TFIM / RBM alpha=3 补训结果

本文件夹是本次 18 组补训结果的独立副本，原始结果未移动或覆盖。

- L=10，alpha=3（30 个隐藏单元），g=1.0、1.5。
- 每个 g 的角度：0°、15°、22.5°、37.5°、45°、60°、67.5°、82.5°、90°。入口使用角度，内部转换为弧度。
- 全部完成：每组 100 个 Optuna trial（含剪枝），保留能量最优的 10 个完整 trial，共 180 个，均训练到第 1000 步；失败 trial 为 0。
- 保留第 5、8、13、21、34、55、89、145、235、381、617、1000 步的模型及优化器 checkpoint，共 4320 个入选 checkpoint 文件，已完成解码验证。

## 文件说明

- `PLAYING_WITH_RBM_COMPLEX/`：本次 18 组完整实验目录，命名和层级与原项目一致。每组包含 SQLite Optuna 数据库、`optuna_trials/TRIAL*/` 的入选结果及原流程保存的恢复/归档数据。
- `summary.csv`：每组状态、搜索 trial 数、完成步数、入选 trial 数和相对路径，方便表格查看。
- `summary.json`：进一步列出每组入选 trial 编号及第 1000 步模型 checkpoint 的相对路径。
- `plots/`：原 `energy_and_infidelity_plot.py` 生成的两个 g 的验证图。图中每个 g 包含全部 13 个角度，因此也使用了此前已有的 4 个角度作比较；此前 8 组实验未复制进本文件夹，其原数据仍留在原项目中。
- `logs/`：原始训练日志、流程说明 `PROTOCOL.md`、环境版本 `environment.txt`、验证记录 `validation_result.json` 等。原日志和原汇总中的绝对路径是来源记录；本文件夹的路径索引请用顶层 `summary.csv` / `summary.json`。
- `scripts/`：本次新增批量训练与验收脚本副本。脚本保留原工作区绝对路径，迁移运行前需调整路径；无需再次训练即可读取现有结果。
- `source_snapshot/`：原项目 Python、Shell 和 README 快照，不含虚拟环境。
- `package_manifest.json`：复制来源与逐文件 SHA256 记录。
- `SHA256SUMS`：本交付文件夹全部文件的校验清单（清单本身除外）。

`optuna_trials` 仅包含筛选后的 10 个 trial；`_preserved_trials` 和 `_recovery` 是保留的归档/恢复材料，不应作为额外入选结果统计。数据库记录的是完整的 100 次搜索。

## 验证

本次原验收确认全部 18 组完成、4320 个入选 checkpoint 可解码、原绘图入口可读取，原有 8 组的 2008 个文件校验值不变。未找到 `rbm_checkpoint.py`，因此未直接运行该入口。复制时逐文件核对源文件与副本 SHA256，并检查每组数据库 100 个 trial、入选 10 个 trial 及最终 checkpoint 存在。

在本文件夹内可再次检查完整性：

```bash
sha256sum -c SHA256SUMS
```

原项目：`/home/tiger/project/basis-dependence-of-RBM-for-TFIM/`

原批量脚本与日志：`/home/tiger/project/logWFE_for_TFIM/`
