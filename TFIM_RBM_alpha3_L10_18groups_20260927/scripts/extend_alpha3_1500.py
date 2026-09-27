#!/usr/bin/env python3
"""Continue all 26 alpha=3 studies' retained trials; immutable original search DBs."""
import os,sys,json,sqlite3,hashlib,fcntl,traceback,argparse
from pathlib import Path
P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM')
OUT=Path('/home/tiger/project/logWFE_for_TFIM/logs/alpha3_extension_1500');OUT.mkdir(exist_ok=True)
def atomic(p,b):
 t=p.with_name(p.name+'.tmp');t.write_bytes(b);t.replace(p)
def tasks():
 out=[]
 for g in [1.,1.5]:
  for a in [i*7.5 for i in range(13)]:
   d=P/'PLAYING_WITH_RBM_COMPLEX'/f'FULLSUM_QIMRBM_COMPLEXNN3.0L10G{g:.1f}ANGLE{a:.1f}'/'CLIPPING0DYNAMIC0NITER1000'
   ts=sorted(d.glob('optuna_trials/TRIAL*'));assert len(ts)==10
   with sqlite3.connect(f'file:{d}/CLIPPING0DYNAMIC0NITER1000.db?mode=ro',uri=True) as c:
    for t in ts:
     n=int(t.name[5:]);params=dict(c.execute('select p.param_name,p.param_value from trial_params p join trials t on t.trial_id=p.trial_id where t.number=?',(n,)).fetchall())
     assert {'learning_rate','diag_shift'}<=params.keys()
     out.append(dict(g=g,angle=a,trial=n,path=str(t),params=params))
 return out
def worker(pilot=False):
 import numpy as np, flax,jax,netket as nk
 sys.path.insert(0,str(P))
 from define_models import create_system
 from NQS_utils import create_optimizer
 assert jax.default_backend()=='gpu'
 done=0
 for task in tasks():
  t=Path(task['path']);meta=t/'extension_1500.json'
  lock=(t/'.extension_1500.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();continue
  try:
   if meta.exists():continue
   params=dict(L=10,W=1,NN=3.,g=task['g'],angle=task['angle'],model='QIM',pbc=False,architecture='RBM_COMPLEX')
   hi,ham,model,_,_=create_system(params)
   opt,shift=create_optimizer(1500,1000,False,False,task['params'])
   state=nk.vqs.FullSumState(hi,model)
   drv=nk.driver.VMC(ham,opt,variational_state=state,preconditioner=nk.optimizer.SR(diag_shift=shift,holomorphic=True))
   recovery=t/'extension_progress.json';start=json.loads(recovery.read_text())['step'] if recovery.exists() else 1000
   state.variables=flax.serialization.from_bytes(state.variables,(t/f'checkpoint_variables_{start}.mpack').read_bytes())
   drv._optimizer_state=flax.serialization.from_bytes(drv._optimizer_state,(t/f'checkpoint_optimizer_{start}.mpack').read_bytes());drv._step_count=start
   initial_energy=float(state.expect(ham).mean.real)
   print('START',task,'step',start,flush=True)
   for end in np.unique(np.geomspace(1000,1500,12,dtype=int))[1:]:
    end=int(end)
    if end<=start:continue
    log=nk.logging.RuntimeLog();drv.run(n_iter=end-start,out=log,show_progress=False)
    assert np.isfinite(float(log.data['Energy']['Mean'][-1].real))
    log.serialize(str(t/f'extension_log_{start}_{end}'))
    atomic(t/f'checkpoint_variables_{end}.mpack',flax.serialization.to_bytes(state.variables))
    atomic(t/f'checkpoint_optimizer_{end}.mpack',flax.serialization.to_bytes(drv._optimizer_state))
    atomic(recovery,json.dumps({'step':end}).encode());start=end
   # Merge original and extension logs without rewriting original history.
   merged=json.loads((t/'run_resume_1000.json').read_text())
   boundaries=[int(x) for x in np.unique(np.geomspace(1000,1500,12,dtype=int))]
   def merge(a,b):
    for k,v in b.items():
     if isinstance(v,dict):merge(a[k],v)
     else:a[k].extend(v)
   for a,b in zip(boundaries[:-1],boundaries[1:]):merge(merged,json.loads((t/f'extension_log_{a}_{b}.json').read_text()))
   assert merged['Energy']['iters']==list(range(1500))
   atomic(t/'run_resume_1500.json',json.dumps(merged).encode())
   # Decode both final artifacts independently before marking complete.
   restored=flax.serialization.from_bytes(state.variables,(t/'checkpoint_variables_1500.mpack').read_bytes())
   flax.serialization.from_bytes(drv._optimizer_state,(t/'checkpoint_optimizer_1500.mpack').read_bytes())
   assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(restored),jax.tree.leaves(state.variables)))
   final_energy=float(state.expect(ham).mean.real)
   atomic(meta,json.dumps(dict(task,status='complete',steps=1500,initial_energy=initial_energy,final_energy=final_energy,checkpoint_steps=boundaries[1:],gpu=os.environ['CUDA_VISIBLE_DEVICES']),indent=2).encode())
   print('DONE',task['g'],task['angle'],task['trial'],initial_energy,final_energy,flush=True)
   done+=1
   if pilot:break
  except Exception:
   print('FAIL',task,flush=True);traceback.print_exc();raise
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 print('WORKER COMPLETE',done,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--inventory',action='store_true');args=ap.parse_args()
 if args.inventory:
  ts=tasks();atomic(OUT/'tasks.json',json.dumps(ts,indent=2).encode())
  baseline={}
  for d in sorted(set(Path(t['path']).parents[1] for t in ts)):
   for f in d.rglob('*'):
    if f.is_file():baseline[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  atomic(OUT/'original_sha256.json',json.dumps(baseline,indent=2).encode());print('tasks',len(ts),'original files',len(baseline))
 else:worker(args.pilot)
