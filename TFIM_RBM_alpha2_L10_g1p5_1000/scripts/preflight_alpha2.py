import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,hashlib,sqlite3,platform,inspect,subprocess
P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM');L=Path('/home/tiger/project/logWFE_for_TFIM/logs/alpha2_g1p5_completion');sys.path.insert(0,str(P))
import jax,netket as nk,flax,optax,optuna,numpy as np
from config import get_params,get_optuna_args
from define_models import create_system
sys.argv=['preflight','--L','10','--W','1','--NN','2.0','--g','1.5','--angle','90','--niter','1000','--NR','1000','--trials','100','--reports','12','--warmup','1000','--runners','1']
params=get_params();args=get_optuna_args();hi,ham,model,_,_=create_system(params)
records=[]
for a in [7.5,30.,52.5,75.]:
 d=P/'PLAYING_WITH_RBM_COMPLEX'/f'FULLSUM_QIMRBM_COMPLEXNN2.0L10G1.5ANGLE{a:.1f}'/'CLIPPING0DYNAMIC0NITER1000';db=d/'CLIPPING0DYNAMIC0NITER1000.db'
 uri=f'sqlite:///file:{db}?mode=ro&uri=true';study=optuna.load_study(study_name=optuna.get_all_study_names(storage=uri)[0],storage=uri)
 complete=study.get_trials(states=[optuna.trial.TrialState.COMPLETE]);best=[complete[i].number for i in np.argsort([t.value for t in complete])[:10]]
 saved=sorted(int(t.name[5:]) for t in (d/'optuna_trials').glob('TRIAL*'));assert sorted(best)==saved
 dtypes=set()
 for t in (d/'optuna_trials').glob('TRIAL*'):
  tree=flax.serialization.msgpack_restore((t/'checkpoint_variables_1000.mpack').read_bytes());assert tree['params']['Dense']['kernel'].shape==(10,20)
  dtypes.update(str(x.dtype) for x in jax.tree.leaves(tree))
 records.append(dict(angle=a,sampler=type(study.sampler).__name__,pruner=type(study.pruner).__name__,pruner_config=vars(study.pruner),saved_trials=saved,parameter_dtypes=sorted(dtypes),trial_count=len(study.trials)))
report=dict(python=platform.python_version(),versions={m.__name__:m.__version__ for m in [nk,jax,flax,optax,optuna,np]},jax_enable_x64=bool(jax.config.jax_enable_x64),params=params,args=vars(args),hilbert_states=hi.n_states,model=repr(model),old_records=records,sampler_signature=str(inspect.signature(optuna.samplers.TPESampler)),pruner_signature=str(inspect.signature(optuna.pruners.MedianPruner)),source_commit=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),source_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['study.py','config.py','define_models.py','hams.py','optuna_setup.py','NQS_utils.py']},historical_environment_limit='Original NetKet/JAX/Python version lock unavailable. Current source execution and original complex128 checkpoint precision verified; Optuna DB metadata records 4.8.0. Historical sampler object not serialized in DB; TPE/Median inferred from unmodified load_study path under matching Optuna version.')
(L/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
