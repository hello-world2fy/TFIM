import os,sys,subprocess,json
from pathlib import Path
L=Path('/home/tiger/project/logWFE_for_TFIM/logs/alpha3_extension_1500')
processes=[]
for gpu in range(8):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 f=(L/f'gpu{gpu}.log').open('a')
 p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('extend_alpha3_1500.py'))],env=env,stdout=f,stderr=subprocess.STDOUT)
 processes.append((gpu,p,f))
result={}
for gpu,p,f in processes:result[str(gpu)]=p.wait();f.close()
(L/'exit_codes.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
sys.exit(any(result.values()))
