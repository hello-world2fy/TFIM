#!/usr/bin/env python3
"""Read-only status, final checkpoint verification and original plot integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from complete_alpha3 import PROJECT, LOGS, ANGLES, SUFFIX, location


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.final:
        import numpy as np
    rows = []
    for g in [1., 1.5]:
        for angle in ANGLES:
            work = PROJECT / location(g, angle)
            row = dict(g=g, angle=angle, status='pending', states={}, steps=0, saved_trials=0, path=str(work))
            db = work / (SUFFIX + '.db')
            if db.exists():
                with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
                    row['states'] = dict(con.execute('SELECT state,COUNT(*) FROM trials GROUP BY state'))
                    best = [r[0] for r in con.execute("SELECT number FROM trials JOIN trial_values USING(trial_id) WHERE state='COMPLETE' ORDER BY value,number LIMIT 10")]
                    if args.final:
                        complete = con.execute("SELECT number,value FROM trials JOIN trial_values USING(trial_id) WHERE state='COMPLETE' ORDER BY number").fetchall()
                        best = [complete[i][0] for i in np.argsort(np.array([r[1] for r in complete]))[:10]]
                paths = list((work / 'optuna_trials').glob('TRIAL*/checkpoint_variables_1000.mpack'))
                row['saved_trials'] = len(paths)
                row['steps'] = 1000 if paths else 0
                row['best_trials'] = best
                validated = LOGS / f'g{g:.1f}_angle{angle:.1f}.json'
                row['status'] = 'complete' if validated.exists() and sum(row['states'].values()) == 100 and len(paths) == 10 else 'in_progress'
                if args.final:
                    assert row['status'] == 'complete', row
                    saved = sorted(int(p.parent.name[5:]) for p in paths)
                    assert saved == sorted(best), row
            rows.append(row)
    (LOGS / 'status_snapshot.json').write_text(json.dumps(rows, indent=2))
    print('g     angle     status         trials/states                  saved  steps')
    for row in rows:
        print(f"{row['g']:<5} {row['angle']:<9} {row['status']:<14} {str(row['states']):<30} {row['saved_trials']:<6} {row['steps']}")
    if args.final:
        import flax.serialization
        import numpy as np
        checkpoint_files = 0
        steps = np.unique(np.geomspace(5, 1000, num=12, dtype=int))
        for row in rows:
            for trial in row['best_trials']:
                trial_path = Path(row['path']) / 'optuna_trials' / f'TRIAL{trial}'
                for step in steps:
                    for kind in ['variables', 'optimizer']:
                        restored = flax.serialization.msgpack_restore((trial_path / f'checkpoint_{kind}_{step}.mpack').read_bytes())
                        if kind == 'variables':
                            assert restored['params']['Dense']['kernel'].shape == (10, 30)
                            for array in restored['params']['Dense'].values():
                                assert np.isfinite(array).all(), (trial_path, step)
                        checkpoint_files += 1
        print(f'All {checkpoint_files} retained variable/optimizer checkpoints decoded successfully')
        old = json.loads((LOGS / 'original_alpha3_sha256.json').read_text())
        for path, digest in old.items():
            assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == digest, f'Original file changed: {path}'
        print(f'Original {len(old)} files: all SHA-256 checks passed')
        (LOGS / 'summary.json').write_text(json.dumps(rows, indent=2))
        lines = ['# alpha=3 completion results', '', '|g|angle (degrees)|status|steps|saved trials|Optuna states|checkpoint directory|', '|---|---|---|---|---|---|---|']
        for row in rows:
            lines.append(f"|{row['g']}|{row['angle']}|{row['status']}|{row['steps']}|{row['saved_trials']}|{row['states']}|{row['path']}/optuna_trials/TRIAL*/checkpoint_variables_1000.mpack|")
        lines += ['', f'All {len(old)} pre-existing alpha=3 files are unchanged (SHA-256).', '', 'Per-study JSON files contain the exact 10 trial numbers and absolute final checkpoint paths. Nonselected/pruned artifacts remain in each study `_preserved_trials/`; recovery snapshots in `_recovery/`.']
        lines += ['', f"Optuna FAIL trials: {sum(row['states'].get('FAIL', 0) for row in rows)}.",
                  '', 'Two duplicate queued launchers for g=1.5, 15 degrees were cancelled before any database/trial was created, then replaced by a first-available-GPU worker. No training trial was interrupted. Details: queue_reassignment.json and PROTOCOL.md.',
                  '', 'The original energy_and_infidelity_plot.py is exercised by --final --plots using read-only SQLite connections. rbm_checkpoint.py was not found in this workspace; checkpoint compatibility is verified via the original Flax/NetKet loading procedure.']
        (LOGS / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    if args.plots:
        assert args.final, '--plots requires --final'
        os.chdir(PROJECT)
        sys.path.insert(0, str(PROJECT))
        import config
        # The original plotting entry point calls setup_workspace even when
        # reading. Replace that I/O factory with a read-only path calculation.
        def readonly_workspace(params, optuna_args, nn_params, root, name):
            path = location(params['g'], params['angle'])
            database = (PROJECT / path / (SUFFIX + '.db')).resolve()
            return str(path), str(path / SUFFIX), f'sqlite:///file:{database}?mode=ro&uri=true'
        config.setup_workspace = readonly_workspace
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        real_save = plt.savefig
        plt.savefig = lambda fname, *a, **kw: real_save(LOGS / Path(fname).name, *a, **kw)
        import energy_and_infidelity_plot
        for g in [1., 1.5]:
            sys.argv = ['energy_and_infidelity_plot.py', '--L', '10', '--NN', '3.0', '--g', str(g), '--NR', '1000']
            energy_and_infidelity_plot.main()
            plt.close('all')
        for path, digest in old.items():
            assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == digest, path
        print('Original energy_and_infidelity_plot.py completed for both g values; original hashes still match')


if __name__ == '__main__':
    main()
