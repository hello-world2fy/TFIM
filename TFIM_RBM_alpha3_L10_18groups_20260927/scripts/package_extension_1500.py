"""Append verified extension results to the existing delivery folder, preserving old data."""
from pathlib import Path
import json,shutil,hashlib,csv
W=Path('/home/tiger/project/logWFE_for_TFIM');P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM');L=W/'logs/alpha3_extension_1500';D=W/'results/TFIM_RBM_alpha3_L10_18groups_20260927'
validation=json.loads((L/'validation.json').read_text());assert validation['trials']==260
copied=0
def copy(src,dst):
 global copied
 if dst.exists():
  assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest(),str(dst)
 else:
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest()
  copied+=1
for d in sorted(set(Path(t['path']).parents[1] for t in json.loads((L/'tasks.json').read_text()))):
 for src in d.rglob('*'):
  if src.is_file() and not src.name.endswith(('.lock','.tmp')):copy(src,D/src.relative_to(P))
for src in L.rglob('*'):
 if src.is_file():copy(src,D/'logs/alpha3_extension_1500'/src.relative_to(L))
for name in ['extend_alpha3_1500.py','launch_extension_1500.py','audit_extension_1500.py','package_extension_1500.py']:
 copy(W/'scripts'/name,D/'scripts'/name)
summary=json.loads((L/'summary_1500.json').read_text())
for r in summary:r['checkpoint_directory']=str(Path(r['checkpoint_directory']).relative_to(P))
(D/'summary_1500.json').write_text(json.dumps(summary,indent=2)+'\n')
with (D/'summary_1500.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
note='''# 更新：全部角度续训至 1500 步\n\n本文件夹沿用原名，现已包含 **g=1.0、1.5 各 13 个角度，共 26 组、260 个保留 trial**。每个 trial 从原第 1000 步参数及优化器状态继续训练 500 步。原先已有的 7.5°、30°、52.5°、75° 结果也已加入本文件夹。\n\n查看 [1500 步结果汇总](summary_1500.csv) 和 [续训验收报告](logs/alpha3_extension_1500/SUMMARY.md)。\n\n实验仍位于原 `PLAYING_WITH_RBM_COMPLEX/.../CLIPPING0DYNAMIC0NITER1000/optuna_trials/TRIAL*/` 中；读取新结果时使用 `checkpoint_variables_1500.mpack` 和 `run_resume_1500.json`。新增中间步骤为 1037、1076、1116、1158、1202、1247、1294、1342、1393、1445、1500。\n\n原数据库、1000 步及之前的 checkpoint 保留。没有重新进行超参数搜索；数据库中的排名和目标值仍是原 1000 步记录。1500 步结果另见每个 trial 的 `extension_1500.json`。原 `summary.csv` / `summary.json` 仍描述第一次 18 组补训，不代表续训后的范围。`package_manifest.json` 是第一次打包的来源记录；当前完整文件校验以重新生成的 `SHA256SUMS` 为准。\n\n以下保留首次交付说明作为历史记录。\n\n---\n\n'''
readme=D/'README.md'
if not readme.read_text().startswith('# 更新：'):readme.write_text(note+readme.read_text())
with (D/'SHA256SUMS').open('w') as f:
 for src in sorted(D.rglob('*')):
  if src.is_file() and src.name!='SHA256SUMS':f.write(hashlib.sha256(src.read_bytes()).hexdigest()+'  '+str(src.relative_to(D))+'\n')
print('New copied files:',copied,'folder:',D)
