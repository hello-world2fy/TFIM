#!/usr/bin/env python3
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,csv,hashlib,sqlite3
import numpy as np,jax,netket as nk,flax
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM');W=Path('/home/tiger/project/logWFE_for_TFIM');L=W/'logs/alpha2_g1p5_completion';sys.path.insert(0,str(P))
from hams import rotated_IsingModel
A=[0.,15.,22.5,37.5,45.,60.,67.5,82.5,90.];steps=[5,8,13,21,34,55,89,145,235,381,617,1000]
baseline=json.loads((L/'protected_sha256.json').read_text())
for f,h in baseline.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for f,h in json.loads((L/'preflight.json').read_text())['source_sha256'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
print('Protected files unchanged:',len(baseline),flush=True)
hi=nk.hilbert.Spin(s=.5,N=10,inverted_ordering=True);assert hi.n_states==1024 and jax.config.jax_enable_x64
states=hi.all_states();model=nk.models.RBM(alpha=2.,param_dtype=complex);template=model.init(jax.random.PRNGKey(0),states);apply=jax.jit(model.apply)
summary=[];metrics=[];decoded=0;new_decoded=0;fig,axs=plt.subplots(4,4,figsize=(16,12))
for idx,a in enumerate([i*7.5 for i in range(13)]):
 d=P/'PLAYING_WITH_RBM_COMPLEX'/f'FULLSUM_QIMRBM_COMPLEXNN2.0L10G1.5ANGLE{a:.1f}'/'CLIPPING0DYNAMIC0NITER1000'
 db=d/'CLIPPING0DYNAMIC0NITER1000.db'
 with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
  counts=dict(c.execute('select state,count(*) from trials group by state').fetchall())
  completed=c.execute("select t.number,v.value from trials t join trial_values v on t.trial_id=v.trial_id where t.state='COMPLETE' and v.objective=0 order by t.number").fetchall()
 assert sum(counts.values())==100 and counts.get('FAIL',0)==0 and counts.get('RUNNING',0)==0,counts
 assert len(completed)>=10
 best=[completed[i] for i in np.argsort([x[1] for x in completed])[:10]]
 assert sorted(n for n,_ in best)==sorted(int(t.name[5:]) for t in (d/'optuna_trials').glob('TRIAL*'))
 ham=rotated_IsingModel(np.deg2rad(a),1.5,10,hi,pbc=False);H=ham.to_sparse();evals,evec=eigsh(H,k=1,which='SA',tol=1e-12,v0=np.random.default_rng(0).normal(size=1024));e0=float(evals[0]);per=[]
 for number,objective in best:
  t=d/'optuna_trials'/f'TRIAL{number}'
  for step in steps:
   for kind in ['variables','optimizer']:
    data=flax.serialization.msgpack_restore((t/f'checkpoint_{kind}_{step}.mpack').read_bytes());decoded+=1;new_decoded+=int(a in A)
    if kind=='variables':
     assert data['params']['Dense']['kernel'].shape==(10,20)
     assert all(x.dtype==np.complex128 and np.all(np.isfinite(x)) for x in jax.tree.leaves(data))
  v=flax.serialization.from_bytes(template,(t/'checkpoint_variables_1000.mpack').read_bytes());lp=np.asarray(apply(v,states));assert lp.shape==(1024,) and np.all(np.isfinite(lp))
  psi=np.exp(lp-lp.real.max());norm=np.linalg.norm(psi);assert np.isfinite(norm) and norm>0;psi/=norm
  energy=float(np.vdot(psi,H@psi).real)
  vs=nk.vqs.FullSumState(hi,model);vs.variables=v;nkenergy=float(vs.expect(ham).mean.real);assert abs(nkenergy-energy)<1e-9
  log=json.loads((t/'run_resume_1000.json').read_text())['Energy'];assert log['iters']==list(range(1000))
  means=log['Mean'];es=np.asarray(means['real'] if isinstance(means,dict) else means);assert np.all(np.isfinite(es)) and abs(float(es[-1])-objective)<1e-10
  row=dict(angle=a,source='new' if a in A else 'existing',trial=number,steps=1000,objective_energy=objective,last_log_iter=999,last_log_energy=float(es[-1]),checkpoint_energy=energy,checkpoint_minus_log=energy-float(es[-1]),relative_energy_error=(energy-e0)/abs(e0),infidelity=float(1-abs(np.vdot(evec[:,0],psi))**2),checkpoint=str((t/'checkpoint_variables_1000.mpack').relative_to(P)))
  metrics.append(row);per.append(row)
  axs.flat[idx].semilogy(log['iters'],np.maximum((es-e0)/abs(e0),1e-16),lw=.8,alpha=.65)
 axs.flat[idx].set_title(f'{a:g} degrees'+(' (existing)' if a not in A else ''));axs.flat[idx].set_xlabel('Iteration');axs.flat[idx].set_ylabel('Relative energy error');axs.flat[idx].grid(alpha=.2)
 summary.append(dict(angle=a,status='complete' if a in A else 'existing_preserved',trial_states=counts,selected_trial_numbers=[n for n,_ in best],steps=1000,saved_trials=10,best_objective_energy=min(x['objective_energy'] for x in per),best_checkpoint_energy=min(x['checkpoint_energy'] for x in per),checkpoint_directory=str((d/'optuna_trials').relative_to(P))))
 print('VALIDATED',a,counts,flush=True)
for ax in axs.flat[13:]:ax.set_visible(False)
fig.suptitle('L=10, W=1, g=1.5, alpha=2, full Hilbert space');fig.tight_layout();fig.savefig(L/'convergence_13_angles.pdf');plt.close(fig)
with (L/'trial_energy_validation.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
(L/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (L/'summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['angle','status','COMPLETE','PRUNED','FAIL','selected_trials','steps','best_objective_energy','best_checkpoint_energy','checkpoint_directory'])
 for r in summary:w.writerow([r['angle'],r['status'],r['trial_states'].get('COMPLETE',0),r['trial_states'].get('PRUNED',0),r['trial_states'].get('FAIL',0),','.join(map(str,r['selected_trial_numbers'])),1000,r['best_objective_energy'],r['best_checkpoint_energy'],r['checkpoint_directory']])
validation=dict(new_studies=9,new_search_trials=900,new_final_models=90,steps=1000,new_decoded_checkpoints=new_decoded,total_decoded_checkpoints=decoded,protected_files_unchanged=len(baseline),coverage_angles=[i*7.5 for i in range(13)],parameter_dtype='complex128',parameter_kernel_shape=[10,20],hilbert_states=1024,jax_enable_x64=bool(jax.config.jax_enable_x64),max_abs_checkpoint_log_difference=max(abs(r['checkpoint_minus_log']) for r in metrics if r['source']=='new'),rbm_checkpoint_py_found=False)
(L/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
lines=['# α=2，g=1.5 补训验收','', '9 个新增角度完成；每组 100 个搜索 trial（包含剪枝），从 COMPLETE 中按原数据库能量排序保留 10 个。新增 90 个最终模型均为 1000 步。已有四个角度未重训。','',f'新增 {new_decoded} 个模型/优化器 checkpoint 已解码；连同原有四组共验证 {decoded} 个。完整 Hilbert 空间为 1024 维，参数为 complex128、权重形状 (10,20)，归一化与能量重算通过。{len(baseline)} 个非目标实验文件校验不变。','', '| 角度 | 状态 | COMPLETE/PRUNED | 入选 trial | 步数 | 最优 checkpoint 能量 |','|---|---|---|---|---|---|']
for r in summary:lines.append(f"| {r['angle']} | {r['status']} | {r['trial_states'].get('COMPLETE',0)}/{r['trial_states'].get('PRUNED',0)} | {','.join(map(str,r['selected_trial_numbers']))} | 1000 | {r['best_checkpoint_energy']:.12f} |")
lines+=['','## 能量记录的步数差异','', 'NetKet 原 driver 在更新前记录能量：日志迭代编号为 0～999，数据库目标值等于最后一条日志（第 999 次更新后的参数能量）；原保存函数在第 1000 次更新完成后写最终 checkpoint。因此重算 checkpoint 能量与末条日志可以不同，相差一次 SGD/SR 更新。逐 trial 的两个能量及差值已原样记录在 trial_energy_validation.csv；未修改数据库或日志。筛选仍按数据库目标能量，不按重算值重新筛选。','', '历史 NetKet/JAX/Python 版本无锁文件，不能宣称环境逐版本或结果逐位复现。已验证当前原源码执行路径、原 Optuna 4.8.0 数据库元信息和 complex128 参数精度；实际配置见 preflight.json 与 actual_angle*.json。原流程未固定搜索/初始化种子。','', '未找到 rbm_checkpoint.py，未直接运行该文件；采用原 Flax/model 模式解码，另用 FullSumState 与稀疏 Hamiltonian 两种方式核对最终能量。','', 'checkpoint_directory 为相对于项目或交付文件夹的路径；各 trial 最终文件为 checkpoint_variables_1000.mpack / checkpoint_optimizer_1000.mpack / run_resume_1000.json。']
(L/'SUMMARY.md').write_text('\n'.join(lines)+'\n');print('PASS',validation,flush=True)
