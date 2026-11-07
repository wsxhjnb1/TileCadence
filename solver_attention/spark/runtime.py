"""Spark-only retained-runtime helpers. Imports do not launch GPU kernels."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def append(path,obj):
    with Path(path).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
def command(argv):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=20)
    return {'argv':argv,'exit_code':r.returncode,'output':r.stdout,'stderr':r.stderr}
def observation():
    return {'utc':utc(),'pid':os.getpid(),
       'compute':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
       'gpu':command(['nvidia-smi','--query-gpu=uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu','--format=csv,noheader']),
       'processes':command(['ps','-eo','pid,ppid,etime,args'])}
def require_slot(obs):
    apps=obs['compute']
    if apps['exit_code']!=0:raise RuntimeError('GPU process census unavailable')
    for line in apps['output'].splitlines():
        if line.strip() and int(line.split(',')[0])!=os.getpid():
            raise RuntimeError('Another compute process is active; retained without interruption: '+line)
    if obs['gpu']['exit_code']!=0 or 'NVIDIA GB10' not in obs['gpu']['output']:
        raise RuntimeError('Expected the authorized Spark GB10')
def hardware():
    p=torch.cuda.get_device_properties(0)
    fields=('name','major','minor','multi_processor_count','total_memory','shared_memory_per_multiprocessor',
            'shared_memory_per_block_optin','regs_per_multiprocessor','max_threads_per_multi_processor','L2_cache_size')
    return {'properties':{k:getattr(p,k,None) for k in fields},'torch':torch.__version__,
            'torch_cuda':torch.version.cuda,'torch_path':torch.__file__,'device_uuid':str(getattr(p,'uuid',None))}
def load_ops(receipt_path):
    receipt=json.loads(Path(receipt_path).read_text())
    if receipt['state']!='complete' or sha(receipt['module_path'])!=receipt['module_sha256']:
        raise RuntimeError('Compiled library receipt/hash mismatch')
    torch.ops.load_library(receipt['module_path'])
    return getattr(torch.ops,receipt['ops_namespace']),receipt
def inputs(n,h,dtype,seed,pattern='random',batch=1):
    torch.manual_seed(seed)
    q,k,v=(torch.randn((batch,n,h,128),dtype=dtype,device='cuda') for _ in range(3))
    if pattern=='zero_q':q.zero_()
    elif pattern=='repeated_k':k=k[:,:1].expand_as(k).contiguous()
    elif pattern=='scaled_q':q.mul_(4)
    return q,k,v
def reference(q,k,v,causal):
    qf,kf,vf=(x.transpose(1,2).float() for x in (q,k,v))
    scores=(qf@kf.transpose(-1,-2))*(128**-.5)
    if causal:
        index=torch.arange(q.shape[1],device=q.device)
        scores.masked_fill_(index[None,:]>index[:,None],-float('inf'))
    return (torch.softmax(scores,-1)@vf).transpose(1,2),torch.logsumexp(scores,-1)
def checks(out,lse,ref,ref_lse,dtype):
    atol,rtol=(.003,.01) if dtype==torch.float16 else (.02,.02)
    x=out.float();y=ref.float();delta=x-y
    good=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.allclose(x,y,atol=atol,rtol=rtol))
    lok=bool(torch.isfinite(lse).all() and torch.allclose(lse,ref_lse,atol=.0005,rtol=.0001))
    def finite(v):
        value=float(v)
        return value if __import__('math').isfinite(value) else None
    return {'ok':good and lok,'output':{'ok':good,'atol':atol,'rtol':rtol,'max_abs':finite(delta.abs().max()),
            'mean_abs':finite(delta.abs().mean())},'lse':{'ok':lok,'atol':.0005,'rtol':.0001,
            'max_abs':finite((lse-ref_lse).abs().max())}}
