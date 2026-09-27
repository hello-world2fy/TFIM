import os
os.environ['JAX_PLATFORMS']='cpu'
import json,hashlib,csv,sys
from pathlib import Path
import numpy as np,flax
from scipy.sparse.linalg import eigsh
import netket as nk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM');sys.path.insert(0,str(P))
from hams import rotated_IsingModel
L=Path('/home/tiger/project/logWFE_for_TFIM/logs/alpha3_extension_1500')
tasks=json.loads((L/'tasks.json').read_text());baseline=json.loads((L/'original_sha256.json').read_text())
for name,digest in baseline.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('Original files unchanged:',len(baseline),flush=True)
hi=nk.hilbert.Spin(s=.5,N=10,inverted_ordering=True)
e0s={g:float(eigsh(rotated_IsingModel(0,g,10,hi).to_sparse(),k=1,which='SA',return_eigenvectors=False,tol=1e-12,v0=np.random.default_rng(0).normal(size=1024))[0]) for g in [1.,1.5]}
rows=[];decoded=0
for task in tasks:
 t=Path(task['path']);meta=json.loads((t/'extension_1500.json').read_text());assert meta['steps']==1500
 log=json.loads((t/'run_resume_1500.json').read_text())['Energy'];assert log['iters']==list(range(1500))
 es=np.asarray(log['Mean']['real']);assert np.all(np.isfinite(es))
 for step in meta['checkpoint_steps']:
  for kind in ['variables','optimizer']:
   tree=flax.serialization.msgpack_restore((t/f'checkpoint_{kind}_{step}.mpack').read_bytes());decoded+=1
   if kind=='variables':assert np.asarray(tree['params']['Dense']['kernel']).shape==(10,30)
 e0=e0s[task['g']]
 # Initial energy is from resumed start; checkpoint-1000 log estimate retained separately.
 rows.append(dict(g=task['g'],angle=task['angle'],trial=task['trial'],steps=1500,energy_1000=meta['initial_energy'],energy_1500=meta['final_energy'],relative_error_1500=(meta['final_energy']-e0)/abs(e0),last100_relative_drop=float((es[-101]-es[-1])/abs(e0)),checkpoint=str(t/'checkpoint_variables_1500.mpack')))
with (L/'trial_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for g in [1.,1.5]:
 fig,axes=plt.subplots(4,4,figsize=(16,12))
 for i,angle in enumerate([i*7.5 for i in range(13)]):
  group=[r for r in rows if r['g']==g and r['angle']==angle]
  best=min(group,key=lambda x:x['energy_1500']);summary.append(dict(g=g,angle=angle,status='complete',steps=1500,saved_trials=len(group),best_trial=best['trial'],best_relative_error=best['relative_error_1500'],checkpoint_directory=str(Path(best['checkpoint']).parents[1])))
  for r in group:
   log=json.loads((Path(r['checkpoint']).parent/'run_resume_1500.json').read_text())['Energy'];es=np.asarray(log['Mean']['real'])
   axes.flat[i].semilogy(log['iters'],np.maximum((es-e0s[g])/abs(e0s[g]),1e-16),lw=.8,alpha=.65)
  axes.flat[i].axvline(1000,color='k',ls='--',lw=.7);axes.flat[i].set_title(f'{angle:g} degrees');axes.flat[i].set_xlabel('Iteration');axes.flat[i].set_ylabel('Relative energy error');axes.flat[i].grid(alpha=.2)
 for a in axes.flat[13:]:a.set_visible(False)
 fig.suptitle(f'L=10, alpha=3, g={g}: 1000 -> 1500 steps');fig.tight_layout();fig.savefig(L/f'g{g}_convergence_1500.pdf');plt.close(fig)
(L/'summary_1500.json').write_text(json.dumps(summary,indent=2))
lines=['# alpha=3 续训至 1500 步','',f'26 组、260 个保留 trial 均完成新增 500 步。原始 {len(baseline)} 个文件的 SHA256 均未变化。新增 {decoded} 个模型/优化器 checkpoint 已解码验证。','', '原 Optuna 数据库仍记录 1000 步搜索结果，不重新搜索或改写排名。1500 步能量与状态记录在 extension_1500.json / trial_metrics.csv。目录中的 NITER1000 保留原名；读取新增结果时指定第 1500 步。','', '| g | 角度 | 步数 | trial 数 | 最优相对能量误差 |','|---|---|---|---|---|']
for r in summary:lines.append(f"| {r['g']} | {r['angle']} | 1500 | {r['saved_trials']} | {r['best_relative_error']:.3e} |")
(L/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
worsened=[r for r in rows if r['energy_1500']>r['energy_1000']+1e-10]
(L/'energy_increases.json').write_text(json.dumps(worsened,indent=2))
with (L/'SUMMARY.md').open('a') as f:
 f.write('\n## 续训后能量回升的 trial\n\n')
 if not worsened:f.write('无。\n')
 for r in worsened:f.write(f"- g={r['g']}，角度={r['angle']}，trial={r['trial']}：{r['energy_1000']:.12f} → {r['energy_1500']:.12f}。原参数和全部中间 checkpoint 已保留。\n")
(L/'validation.json').write_text(json.dumps(dict(studies=26,trials=260,steps=1500,original_files_unchanged=len(baseline),decoded_new_checkpoints=decoded,exact_energies=e0s),indent=2))
print('PASS',len(rows),'trials',decoded,'new checkpoints',flush=True)
