#!/usr/bin/env python3
"""Complete the original 100-trial TFIM studies, one process per GPU.

Run with the basis-dependence-of-RBM-for-TFIM/.venv/bin/python interpreter.
Default: validate g=1 angle=0 first, then dispatch the other 17 studies.
Original model/optimizer/helpers are imported directly; the objective below
adds durable chunk-boundary recovery and preserves discarded trial artifacts.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

PROJECT = Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM')
LOGS = Path(__file__).resolve().parents[1] / 'logs' / 'alpha2_g1p5_completion'
ANGLES = [0., 15., 22.5, 37.5, 45., 60., 67.5, 82.5, 90.]
SUFFIX = 'CLIPPING0DYNAMIC0NITER1000'


def location(g, angle):
    return Path('PLAYING_WITH_RBM_COMPLEX') / f'FULLSUM_QIMRBM_COMPLEXNN2.0L10G{g:.1f}ANGLE{angle:.1f}' / SUFFIX


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(g, angle):
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS / f'g{g:.1f}_angle{angle:.1f}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    validated = LOGS / f'g{g:.1f}_angle{angle:.1f}.json'
    if validated.exists():
        result = json.loads(validated.read_text())
        if (result.get('total_trials') == 100 and result.get('saved_trials') == 10
                and all(Path(x['checkpoint']).is_file() for x in result['checkpoints'])):
            print(f'SKIP validated complete g={g} angle={angle}', flush=True)
            return
    # Acquire before importing JAX, so even overlapping launchers cannot put
    # more than one training process on a physical GPU or repeat a study.
    pool = os.environ.get('ALPHA3_GPU_POOL')
    if pool:
        gpu_lock = None
        while gpu_lock is None:
            for candidate in pool.split(','):
                handle = (LOGS / f'gpu{candidate}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                else:
                    gpu, gpu_lock = candidate, handle
                    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
                    break
            if gpu_lock is None:
                time.sleep(1)
    else:
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        gpu_lock = (LOGS / f'gpu{gpu}.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    print(f'ACQUIRED physical GPU {gpu} for g={g} angle={angle}', flush=True)
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import flax
    import jax
    import netket as nk
    import optuna
    # History.__getattr__ recurses during pickle's __setstate__ lookup before
    # _value_dict exists. Explicit state restoration fixes only deserialization.
    def restore_history(self, state):
        slots = state[1] if isinstance(state, tuple) else state
        for key, value in slots.items():
            object.__setattr__(self, key, value)
    nk.utils.history.History.__setstate__ = restore_history
    from config import get_params, get_optuna_args, ALL_NN_PARAMS
    from define_models import create_system
    from optuna_setup import setup_workspace
    from NQS_utils import generate_hyperparameters, create_checkpoint_callback, save_checkpoint

    sys.argv = [sys.argv[0], '--L', '10', '--NN', '2.0', '--g', str(g), '--angle', str(angle),
                '--niter', '1000', '--NR', '1000', '--trials', '100', '--reports', '12', '--runners', '1']
    params, args = get_params(), get_optuna_args()
    hi, ham, model, _, name = create_system(params)
    work, study_name, storage = setup_workspace(params, args, ALL_NN_PARAMS[params['architecture']], params['ROOT_DIR'], name)
    work = Path(work)
    # Deliberately match study.py: load_study defaults to TPE, not the
    # non-persisted RandomSampler passed when the study was created.
    study = optuna.load_study(study_name=study_name, storage=storage)
    print('DEVICES', jax.devices(), 'SAMPLER', type(study.sampler).__name__, flush=True)
    assert jax.default_backend() == 'gpu', 'GPU backend required'
    assert jax.config.jax_enable_x64 and hi.n_states == 1024
    atomic(LOGS / f'actual_angle{angle:.1f}.json', json.dumps({'params': params, 'args': vars(args), 'jax_enable_x64': bool(jax.config.jax_enable_x64), 'hilbert_states': hi.n_states, 'sampler': type(study.sampler).__name__, 'pruner': type(study.pruner).__name__, 'pruner_config': vars(study.pruner), 'gpu': gpu}, indent=2).encode())
    checkpoints = np.unique(np.geomspace(5, 1000, num=12, dtype=int))
    bounds = dict(learning_rate=[1e-5, 1e-2], diag_shift=[1e-5, 1e-1], max_step_size=[1e-5, 1e-1])

    def objective(trial):
        trial_dir = work / 'optuna_trials' / f'TRIAL{trial.number}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        restart = work / '_recovery' / f'TRIAL{trial.number}'
        restart.mkdir(parents=True, exist_ok=True)
        optimizer, shift = generate_hyperparameters(trial, 1000, 1000, False, False, bounds)
        state = nk.vqs.FullSumState(hi, model)
        driver = nk.driver.VMC(ham, optimizer, variational_state=state,
                               preconditioner=nk.optimizer.SR(diag_shift=shift, holomorphic=True))
        log, start = nk.logging.RuntimeLog(), 0
        if (restart / 'latest.json').exists():
            meta = json.loads((restart / 'latest.json').read_text())
            start = meta['step']
            state.variables = flax.serialization.from_bytes(state.variables, (restart / f'variables_{start}.mpack').read_bytes())
            driver._optimizer_state = flax.serialization.from_bytes(driver._optimizer_state, (restart / f'optimizer_{start}.mpack').read_bytes())
            driver._step_count = start
            log = pickle.loads((restart / f'log_{start}.pickle').read_bytes())
            print(f'RESUME trial={trial.number} step={start}', flush=True)
        elif any(trial_dir.iterdir()):
            raise RuntimeError(f'Existing partial trial has no durable recovery manifest: {trial_dir}; preserved without overwriting')

        base_callback = create_checkpoint_callback(str(trial_dir), checkpoints, 12)
        def callback(step, data, drv):
            if (trial_dir / f'checkpoint_variables_{step}.mpack').exists():
                return True
            return base_callback(step, data, drv)

        def archive_pruned():
            target = work / '_preserved_trials' / f'TRIAL{trial.number}'
            target.parent.mkdir(exist_ok=True)
            if trial_dir.exists():
                if target.exists() and not any(trial_dir.iterdir()):
                    trial_dir.rmdir()  # crash after archive but before study.tell
                else:
                    trial_dir.rename(target)

        if start:
            trial.report(float(log.data['Energy']['Mean'][-1].real), start)
            if trial.should_prune():
                archive_pruned()
                raise optuna.TrialPruned()
        for current in checkpoints:
            current = int(current)
            if current <= start:
                continue
            driver.run(obs={}, n_iter=current-start, out=log, callback=[callback], show_progress=False)
            # Manifest written last: a crash can never select a partial snapshot.
            atomic(restart / f'variables_{current}.mpack', flax.serialization.to_bytes(state.variables))
            atomic(restart / f'optimizer_{current}.mpack', flax.serialization.to_bytes(driver._optimizer_state))
            atomic(restart / f'log_{current}.pickle', pickle.dumps(log))
            atomic(restart / 'latest.json', json.dumps({'step': current}).encode())
            start = current
            score = float(log.data['Energy']['Mean'][-1].real)
            trial.report(score, current)
            if trial.should_prune():
                archive_pruned()
                raise optuna.TrialPruned()
        log.serialize(str(trial_dir / 'run_resume_1000'))
        value = float(log.data['Energy']['Mean'][-1].real)
        if not np.isfinite(value):
            raise optuna.TrialPruned('Trial diverged resulting in NaN energy.')
        if not (trial_dir / 'checkpoint_variables_1000.mpack').exists():
            save_checkpoint(str(trial_dir), driver, 1000)
        return value

    # A killed process leaves RUNNING trials. Continue that same trial and its
    # optimizer; do not create a new trial or reset its learned parameters.
    for frozen in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING]):
        live = optuna.trial.Trial(study, frozen._trial_id)
        try:
            value = objective(live)
        except optuna.TrialPruned:
            study.tell(live, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(live, value)
    remaining = 100 - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1)

    completed = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    # Match the original plotter, including NumPy's ordering of exact ties.
    best = [completed[i] for i in np.argsort(np.array([t.value for t in completed]))[:10]]
    assert len(best) == 10, f'Only {len(best)} complete trials; all data preserved'
    keep = {t.number for t in best}
    archive = work / '_preserved_trials'
    archive.mkdir(exist_ok=True)
    for path in (work / 'optuna_trials').glob('TRIAL*'):
        if int(path.name[5:]) not in keep:
            path.rename(archive / path.name)

    # Same Flax template/load/model.apply path used by the original plotter.
    template = model.init(jax.random.PRNGKey(0), hi.all_states())
    results = []
    for trial in best:
        path = work / 'optuna_trials' / f'TRIAL{trial.number}'
        for step in checkpoints:
            for kind in ['variables', 'optimizer']:
                assert (path / f'checkpoint_{kind}_{step}.mpack').is_file(), (path, kind, step)
        variables = flax.serialization.from_bytes(template, (path / 'checkpoint_variables_1000.mpack').read_bytes())
        assert variables['params']['Dense']['kernel'].shape == (10, 20)
        assert np.all(np.isfinite(np.asarray(model.apply(variables, hi.all_states()))))
        history = json.loads((path / 'run_resume_1000.json').read_text())
        assert len(history['Energy']['iters']) == 1000
        results.append({'trial': trial.number, 'steps': 1000, 'checkpoint': str((PROJECT / path / 'checkpoint_variables_1000.mpack').resolve())})
    result = {'g': g, 'angle_degrees': angle, 'status': 'complete', 'total_trials': len(study.trials),
              'complete_trials': len(completed), 'saved_trials': 10, 'steps': 1000, 'checkpoints': results,
              'sampler': type(study.sampler).__name__, 'pruner': type(study.pruner).__name__,
              'versions': {x: __import__(x).__version__ for x in ['jax', 'netket', 'optuna', 'flax']}}
    atomic(LOGS / f'g{g:.1f}_angle{angle:.1f}.json', json.dumps(result, indent=2).encode())
    print('VALIDATED', json.dumps(result), flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=float);args=parser.parse_args()
    if args.worker is not None:
        assert args.worker in ANGLES
        worker(1.5,args.worker);return
    processes=[];stamp=time.strftime('%Y%m%d_%H%M%S');commands=[]
    for angle in ANGLES:
        env=dict(os.environ,ALPHA3_GPU_POOL='0,1,2,3,4,5,6,7',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(angle)]
        commands.append({'command':command,'environment':{k:env[k] for k in ['ALPHA3_GPU_POOL','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}})
        log=(LOGS/f'angle{angle:.1f}_{stamp}.log').open('w')
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((angle,p,log));print('QUEUED',angle,flush=True)
    atomic(LOGS/f'commands_{stamp}.json',json.dumps(commands,indent=2).encode())
    result={}
    for angle,p,log in processes:result[str(angle)]=p.wait();log.close()
    atomic(LOGS/f'exit_codes_{stamp}.json',json.dumps(result,indent=2).encode());print(result,flush=True)
    sys.exit(any(result.values()))
if __name__=='__main__':main()
