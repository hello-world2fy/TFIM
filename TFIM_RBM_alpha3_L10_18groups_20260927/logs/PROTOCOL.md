# alpha=3 completion protocol

Source repository commit: 77c3ab0 (original Python sources unchanged).
18 new studies: L=10, alpha=3 (30 hidden units), g=1.0/1.5, degrees 0,15,22.5,37.5,45,60,67.5,82.5,90.
Original 8 studies: 100 trials each, saved directories exactly lowest-energy 10 COMPLETE trials. Original file SHA-256 recorded separately.

- Complex NetKet RBM; FullSumState; open boundaries; holomorphic SR; SGD.
- CLI angle is degrees; create_system converts degrees*pi/180 exactly once.
- learning_rate log-uniform bounds 1e-5..1e-2; diag_shift 1e-5..1e-1.
- No clipping, no decay; niter=NR=1000; 12 geomspace report/checkpoint steps: 5,8,13,21,34,55,89,145,235,381,617,1000.
- MedianPruner defaults. setup_workspace creates RandomSampler but study.py reloads without sampler, thus actual optimization uses default TPESampler. Preserve that behavior.
- No fixed seed in original sampler or FullSumState; preserve unseeded initialization. Seed 0 is used only for deserialization template, as in the original plotter.
- 100 total Optuna trials, including pruned trials; only COMPLETE trials run all 1000 steps. Rank by final reported energy, save best 10 in optuna_trials.
- For preservation, nonselected and pruned trial files are moved into _preserved_trials rather than deleted.
- The imported original model, Hamiltonian, optimizer, callback and save functions are unchanged. Wrapper adds atomic _recovery snapshots after each report chunk, suppresses progress bars, and makes same-trial interrupted recovery possible.
- One worker per GPU; first g=1 angle=0 must finish and validate before remaining 17 start.
- Runtime dependencies are recorded in environment.txt. The original repository has no pinned NetKet/JAX environment; Optuna 4.8.0 matches existing DB metadata. NetKet 3.20.5 and JAX 0.7.2 provide compatible APIs. Python 3.12 is required by existing plotter f-string syntax.
- rbm_checkpoint.py and an existing logWFE_for_TFIM checkout were not found on this machine; scripts placed in /home/tiger/project/logWFE_for_TFIM/scripts. Compatibility checked using original Flax/model reading procedure.

Recovery validation: restoring original optimizer/parameters/log at step 5 and continuing to step 8 reproduces uninterrupted parameters exactly (maximum difference 0.0). NetKet History pickle compatibility is handled by explicit slot restoration in the wrapper.

Scheduling update: after independently validating all 12 checkpoint pairs for pilot TRIAL0 and exact recovery, started the other seven GPUs via --remaining-only --gpus 1,2,3,4,5,6,7 while the pilot continues its 100-trial search on GPU0. Per-study and per-GPU file locks serialize overlapping launcher queues; already validated studies are skipped without additional training.

Queue reassignment: two duplicate launchers waiting to start g=1.5 angle=15 were cancelled before NetKet import or database creation (RSS below 17 MB). The replacement worker uses ALPHA3_GPU_POOL=0,1,2,3,4,5,6,7 and acquires the first free GPU lock. No training trial was interrupted or failed. Old coordinator exit statuses may reflect this scheduling cancellation; queue_reassignment.json records both PIDs and log files. Final database audits determine actual training completion.

Final verification command (run after all 18 status rows are complete):

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLBACKEND=Agg /home/tiger/project/basis-dependence-of-RBM-for-TFIM/.venv/bin/python /home/tiger/project/logWFE_for_TFIM/scripts/audit_alpha3.py --final --plots
```

This decodes all 4320 retained checkpoint files, checks NumPy top-10 selection, verifies the 2008 original file hashes, writes SUMMARY.md and summary.json, and invokes the original energy_and_infidelity_plot.py for both g values with a read-only workspace factory. Original plotting source is unchanged; generated PDFs go into this log directory.

Resume/idempotent completion command:

```sh
/home/tiger/project/basis-dependence-of-RBM-for-TFIM/.venv/bin/python /home/tiger/project/logWFE_for_TFIM/scripts/complete_alpha3.py
```

Already validated studies are skipped. For a specific unfinished group use --worker --g VALUE --angle DEGREES with CUDA_VISIBLE_DEVICES set, or ALPHA3_GPU_POOL=0,1,2,3,4,5,6,7 for the first available GPU. All model training remains sequential within each study.
