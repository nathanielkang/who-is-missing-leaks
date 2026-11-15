import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, numpy as np, json, time
sys.path.insert(0,HERE)
from core import *
C=np.load(HERE + '/data/nhanes_counts.npy'); psus=np.load(HERE + '/data/nhanes_psu.npy')
strata=psus//10; ust=np.unique(strata)
B=int(sys.argv[1]); rng=np.random.default_rng(20261115)
res={k:[] for k in ['e0','e1','joint','prod','diff','hid','hidc']}
t0=time.time()
for b in range(B):
    idx=[]
    for s in ust:
        m=np.where(strata==s)[0]; idx+=list(rng.choice(m,size=len(m),replace=True))
    tot=C[idx].sum(0); pooled=tot.sum(0); n=pooled.sum(1)
    g=np.array([margins(pooled[h]/n[h]) for h in range(2)])
    res['e1'].append(np.abs(np.log(g[1]/g[0]))); res['e0'].append(np.abs(np.log((1-g[1])/(1-g[0]))))
    j,p=occ_levels(pooled[0]/n[0],pooled[1]/n[1])   # patterns observed in both groups, same set for joint and product
    res['joint'].append(j); res['prod'].append(p); res['diff'].append(j-p)
    if b<int(sys.argv[2]):
        res['hid'].append(hidden_joint_level(pooled[0]/n[0],pooled[1]/n[1],.5,.5)); res['hidc'].append(hidden_closed(g[0],g[1],.5,.5))
out={}
for k,v in res.items():
    v=np.array(v); out[k]={'lo':np.percentile(v,2.5,axis=0).tolist(),'hi':np.percentile(v,97.5,axis=0).tolist(),'n':len(v)}
out['diff_pos']=float((np.array(res['diff'])>0).mean())
json.dump(out,open(HERE + '/results/out_nh_boot.json','w'),indent=1)
print(json.dumps(out,indent=1)); print('time',time.time()-t0)
