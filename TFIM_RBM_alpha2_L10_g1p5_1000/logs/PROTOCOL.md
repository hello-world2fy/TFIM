# alpha=2, g=1.5 reproduction protocol

## Scope and evidence

L=10, W=1, NN=2.0 (20 hidden units), g=1.5. New degrees: 0,15,22.5,37.5,45,60,67.5,82.5,90. Existing degrees 7.5,30,52.5,75 preserved. Initial angle0 directory contained a zero-trial Optuna 4.8.0 database; preserved backup in preexisting_targets and resumed same database. All other eight target directories were absent. Protected hashes include every pre-existing experiment file outside these nine target directories, not just alpha=2.

Source commit 77c3ab0c31e3bd4229544774c1f1f1389c21eda6. The six requested original Python files remain unchanged; hashes and snapshots are included. Historical original Python/NetKet/JAX versions are not pinned in the source repository, so exact historical environment identity cannot be proved. Optuna metadata in all four old databases is 4.8.0. All 40 old final models decode with complex128 parameters and Dense kernel (10,20). The current environment is recorded in preflight.json, environment.txt and package_direct_urls.json (extracted with importlib.metadata because this venv has no pip module).

## Actual training

- Original create_system, rotated_IsingModel, create_optimizer, generate_hyperparameters, checkpoint callback and serialization helpers imported directly.
- QIM: open chain, J=1, unchanged rotation x'=cos(theta)x+sin(theta)z, z'=cos(theta)z-sin(theta)x. CLI uses degrees, converted once internally.
- Spin(1/2,N=10,inverted_ordering=True), full 1024-state Hilbert space, including at 90 degrees. No symmetry sector restriction, masking or ED training targets.
- Complex RBM(alpha=2.0), original default biases/initializers/activation; FullSumState, VMC, holomorphic SR, Optax SGD. x64 true, parameters complex128.
- No clipping and no dynamic decay. learning_rate log search [1e-5,1e-2], diag_shift log search [1e-5,1e-1]. No imported best hyperparameters.
- niter=NR=1000, trials=100 total including pruning, reports=12, warmup=1000, runners=1 within each study.
- Reports and original checkpoints at 5,8,13,21,34,55,89,145,235,381,617,1000. trial.should_prune is called after every report, including step1000.
- setup_workspace creates a RandomSampler/MedianPruner study, then the original load_study call without explicit sampler/pruner produces TPESampler and MedianPruner under Optuna4.8.0. The database does not persist sampler/pruner Python objects. This reproduces the evidenced source execution path; not a claim to recover an otherwise undocumented historical sampler object.
- TPESampler defaults: n_startup_trials=10, n_ei_candidates=24, multivariate=False, seed=None. Full signature recorded in preflight.json. MedianPruner defaults: n_startup_trials=5, n_warmup_steps=0, interval_steps=1, n_min_trials=1. CLI warmup1000 is not wired to pruner warmup in the original source.
- No fixed training or sampler seed, matching source. Seed0 used solely to initialize a deserialization template in validation, never as training initialization. Runs are not claimed bitwise reproducible.
- Sequential trials within each study; up to8 distinct studies on8 GPUs. Ninth study queues for a free GPU. Hardware allocation and actual params recorded in actual_angle*.json; launch commands/environment in commands_*.json.

## Recovery and preservation

The new complete_alpha2_g1p5.py adapts the already validated scheduling/recovery wrapper complete_alpha3.py solely to NN=2.0, g=1.5 scope, shape checks and log directory. It mirrors original objective updates/report/pruning sequence while importing the original model/Hamiltonian/optimizer/save helpers. Original source files were not edited. Recovery writes parameters, optimizer and runtime log after each report chunk, then atomically advances latest.json. RUNNING trial is resumed with the same trial ID and parameters; search runs only 100 minus existing total trial count. Non-selected/pruned trial artifacts are archived in _preserved_trials rather than deleted. Median/TPE in-memory RNG state is not serialized, consistent with unseeded original load_study; an interrupted restart is not promised bitwise identical.

Top10 selection uses the original NumPy argsort over COMPLETE trial objective values. Fewer than10 COMPLETE trials causes a visible failure, not extra search. Data outside target studies are fingerprinted for final validation.

## Run / safely resume

```bash
/home/tiger/project/basis-dependence-of-RBM-for-TFIM/.venv/bin/python /home/tiger/project/logWFE_for_TFIM/scripts/complete_alpha2_g1p5.py
```

Each completed study is skipped. Final audit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/tiger/project/basis-dependence-of-RBM-for-TFIM/.venv/bin/python /home/tiger/project/logWFE_for_TFIM/scripts/audit_alpha2_g1p5.py
```

rbm_checkpoint.py was not found under the project/opt paths. Compatibility is tested with the original Flax/model loading convention; direct execution of that missing file is not claimed. NetKet's iter generator yields before update_parameters: logs0..999 are pre-update values, while the explicit final checkpoint contains1000 parameter updates. Both energies and their difference are kept in the audit CSV without modifying logs/database.
