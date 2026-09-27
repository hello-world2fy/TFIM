from pathlib import Path
import sqlite3,json
P=Path('/home/tiger/project/basis-dependence-of-RBM-for-TFIM/PLAYING_WITH_RBM_COMPLEX')
rows=[]
for a in [0.,15.,22.5,37.5,45.,60.,67.5,82.5,90.]:
 d=P/f'FULLSUM_QIMRBM_COMPLEXNN2.0L10G1.5ANGLE{a:.1f}'/'CLIPPING0DYNAMIC0NITER1000';db=d/'CLIPPING0DYNAMIC0NITER1000.db'
 if not db.exists():continue
 with sqlite3.connect(f'file:{db}?mode=ro',uri=True,timeout=30) as c:
  counts=dict(c.execute('select state,count(*) from trials group by state').fetchall());run=c.execute("select number from trials where state='RUNNING'").fetchall()
 steps={}
 for (n,) in run:
  f=d/'_recovery'/f'TRIAL{n}'/'latest.json'
  steps[n]=json.loads(f.read_text())['step'] if f.exists() else 0
 rows.append(dict(angle=a,states=counts,running_steps=steps))
print(json.dumps(rows))
L=Path('/home/tiger/project/logWFE_for_TFIM/logs/alpha2_g1p5_completion')
print('errors',[p.name for p in L.glob('*.log') if 'Traceback' in p.read_text()])
