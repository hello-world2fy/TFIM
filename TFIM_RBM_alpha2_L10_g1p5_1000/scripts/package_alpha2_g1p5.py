from pathlib import Path
import json,hashlib,shutil
W=Path('/home/tiger/project/logWFE_for_TFIM');P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM');L=W/'logs/alpha2_g1p5_completion';D=W/'results/TFIM_RBM_alpha2_L10_g1p5_1000'
v=json.loads((L/'validation.json').read_text());assert v['new_final_models']==90
D.mkdir(exist_ok=True);count=0
def copy(s,t):
 global count
 if t.exists():assert hashlib.sha256(s.read_bytes()).digest()==hashlib.sha256(t.read_bytes()).digest(),str(t)
 else:
  t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,t);assert hashlib.sha256(s.read_bytes()).digest()==hashlib.sha256(t.read_bytes()).digest();count+=1
for a in [i*7.5 for i in range(13)]:
 d=P/'PLAYING_WITH_RBM_COMPLEX'/f'FULLSUM_QIMRBM_COMPLEXNN2.0L10G1.5ANGLE{a:.1f}'
 for s in d.rglob('*'):
  if s.is_file() and not s.name.endswith(('.lock','.tmp')):copy(s,D/s.relative_to(P))
for s in L.rglob('*'):
 if s.is_file() and not s.name.endswith('.lock'):copy(s,D/'logs'/s.relative_to(L))
for name in ['preflight_alpha2.py','complete_alpha2_g1p5.py','status_alpha2_g1p5.py','audit_alpha2_g1p5.py','package_alpha2_g1p5.py']:
 copy(W/'scripts'/name,D/'scripts'/name)
for name in ['summary.json','summary.csv','SUMMARY.md']:
 copy(L/name,D/name)
(D/'README.md').write_text('''# TFIM complex RBM：α=2，L=10，W=1，g=1.5\n\n新增9个角度已完成：0°、15°、22.5°、37.5°、45°、60°、67.5°、82.5°、90°。每个角度100个Optuna trial（含剪枝），保留最优10个完整trial，各1000步。没有加训到1500步。\n\n本交付文件夹同时收录原有7.5°、30°、52.5°、75°的未改动副本，共覆盖0°～90°、间隔7.5°的全部13个角度。\n\n- [验收汇总](SUMMARY.md)\n- [逐角度表格](summary.csv)\n- [运行协议及证据限制](logs/PROTOCOL.md)\n- [逐trial能量核对](logs/trial_energy_validation.csv)\n- [13角度训练曲线](logs/convergence_13_angles.pdf)\n\n`PLAYING_WITH_RBM_COMPLEX/` 保持原命名层级、数据库和 checkpoint 格式。`optuna_trials` 为入选10个trial；`_preserved_trials` 和 `_recovery` 是额外保留的归档/恢复数据，不计入入选数量。\n\n`summary.csv` / `summary.json` 中路径相对于本文件夹；原日志中的绝对路径为来源记录。`logs/` 保存完整训练日志、配置、环境版本、源码快照及校验记录，`scripts/` 是任务调度/恢复/验收脚本（保留原工作区路径，迁移运行前需调整）。\n\n历史NetKet/JAX/Python版本未锁定，不能宣称环境逐版本一致或逐位复现；实际采用原源码训练/搜索流程、相同预算与精度。原流程未固定训练随机种子。末条日志与最终参数相差一次更新，能量差异已单独记录，原始数据未改写。\n\n可在本目录运行 `sha256sum -c SHA256SUMS` 验证完整性。\n''')
with (D/'SHA256SUMS').open('w') as f:
 for s in sorted(D.rglob('*')):
  if s.is_file() and s.name!='SHA256SUMS':f.write(hashlib.sha256(s.read_bytes()).hexdigest()+'  '+str(s.relative_to(D))+'\n')
print('Copied',count,'files;',D)
