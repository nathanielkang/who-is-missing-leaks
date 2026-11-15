import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, json, numpy as np
sys.path.insert(0,HERE)
from core import *; from lps import transport_zero
cells=json.load(open(HERE + '/data/acs2021_ca_adult_cells.json'))
RESTRICT = (sys.argv[1]=='reported'); B=int(sys.argv[2]); SEED=20261115
# population counts POP[band,h,pat]; field order: wage, person income, education, employer insurance; R_k = 1 - flag
POP=np.zeros((3,2,16)); nalloc=np.zeros(2); ntot=np.zeros(2)
for key,v in cells.items():
    b,cit,fc,fw,fp,fs,fh=map(int,key.split('|')); h=int(cit==5); ntot[h]+=v
    if fc==1: nalloc[h]+=v
    if RESTRICT and fc==1: continue
    r=(1-fw)|((1-fp)<<1)|((1-fs)<<2)|((1-fh)<<3)
    POP[b,h,r]+=v
print('adults',ntot.sum(),'noncit',ntot[1],'alloc citizenship',nalloc, 'share alloc',nalloc.sum()/ntot.sum())
print('population used',POP.sum(),'by group',POP.sum((0,2)))
def analyse(T, verbose=False):
    """T[band,h,pat] counts."""
    pooled=T.sum(0); n=pooled.sum(1); P=pooled/n[:,None]
    g=np.array([margins(P[h]) for h in range(2)])
    out={}
    out['rate']=margins(pooled.sum(0)/n.sum()); out['g1']=g[1]; out['g0']=g[0]
    out['e1']=np.abs(np.log(g[1]/g[0])); out['e0']=np.abs(np.log((1-g[1])/(1-g[0])))
    Pb=T/T.sum(2,keepdims=True); gb=np.array([[margins(Pb[b,h]) for h in range(2)] for b in range(3)])
    out['band']=np.maximum(np.abs(np.log((1-gb[:,1])/(1-gb[:,0]))),np.abs(np.log(gb[:,1]/gb[:,0]))).max(0)
    out['band_e0']=np.abs(np.log((1-gb[:,1])/(1-gb[:,0]))).max(0)
    out['joint'],out['prod']=occ_levels(P[0],P[1])
    out['joint_strict']=level(P[0],P[1])
    jb=[level_occupied(Pb[b,0],Pb[b,1]) for b in range(3)]; out['joint_band']=max(jb); out['joint_bands']=jb
    out['prod_band']=max(level(prod_from_margins(gb[b,0]),prod_from_margins(gb[b,1])) for b in range(3))
    if verbose:
        out['empty_both']=int(((pooled[0]==0)&(pooled[1]==0)).sum()); out['empty_one']=int(((pooled[0]==0)^(pooled[1]==0)).sum())
        r=int(np.argmax(np.where((P[0]>0)&(P[1]>0),np.abs(np.log(np.where(P[0]>0,P[0],1)/np.where(P[1]>0,P[1],1))),-1))); out['argmax']=(r,PAT[r].tolist(),pooled[:,r].tolist())
        bb=int(np.argmax(jb)); m=(Pb[bb,0]>0)&(Pb[bb,1]>0); rr=int(np.argmax(np.where(m,np.abs(np.log(np.where(m,Pb[bb,0],1)/np.where(m,Pb[bb,1],1))),-1))); out['band_argmax']=(bb,rr,PAT[rr].tolist(),T[bb,:,rr].tolist())
        out['hid_lp']=hidden_joint_level(P[0],P[1],.5,.5); out['hid_closed']=hidden_closed(g[0],g[1],.5,.5)
        out['hid_lp_band']=[hidden_joint_level(Pb[b,0],Pb[b,1],.5,.5) for b in range(3)]
        out['hid_closed_band']=[hidden_closed(gb[b,0],gb[b,1],.5,.5) for b in range(3)]
        out['hid1']=[hidden_closed([g[0][k]],[g[1][k]],.5,.5) for k in range(4)]
        v,q,pi=transport_zero(P[0],P[1]); out['tr_pooled_fields']=v; out['tr_pooled_task']=q[15]
        out['nosupp_fields_pooled']=g.min(0).sum(); out['obs_per_adult']=float((pooled.sum(0)@SIZE)/n.sum()); out['obs_per_adult_g']=[float(P[h]@SIZE) for h in range(2)]
        wb=T.sum((1,2))/T.sum(); trb=[transport_zero(Pb[b,0],Pb[b,1]) for b in range(3)]
        out['tr_band_fields']=float(sum(wb[b]*trb[b][0] for b in range(3))); out['tr_band_task']=float(sum(wb[b]*trb[b][1][15] for b in range(3)))
        out['nosupp_fields_band']=float(sum(wb[b]*gb[b].min(0).sum() for b in range(3))); out['nosupp_task_band']=float(sum(wb[b]*min(Pb[b,0,15],Pb[b,1,15]) for b in range(3)))
        # per adult retained under pooled transport: E|T| = same in both groups
        out['wb']=wb.tolist(); out['n_bh']=T.sum(2).tolist()
    return out
def show(o):
    for k,v in o.items(): print('  ',k, np.round(v,4) if not isinstance(v,(tuple,list,int)) else v)
print('=== full population file ==='); show(analyse(POP,True))
# stratified extract: 10,000 non-citizens + 30,000 citizens, simple random sampling without replacement
rng=np.random.default_rng(SEED)
S=np.zeros_like(POP)
for h,nh in ((0,30000),(1,10000)):
    flat=POP[:,h,:].ravel().astype(np.int64); S[:,h,:]=rng.multivariate_hypergeometric(flat,nh).reshape(3,16)
np.save(HERE + '/data/acs_extract_%s.npy'%sys.argv[1],S)
print('=== extract n=40,000 ==='); o=analyse(S,True); show(o)
# bootstrap within citizenship groups
res={'diff':[],'joint':[],'prod':[],'e0':[],'band':[]}
for i in range(B):
    T=np.zeros_like(S)
    for h in range(2):
        flat=S[:,h,:].ravel(); T[:,h,:]=rng.multinomial(int(flat.sum()),flat/flat.sum()).reshape(3,16)
    a=analyse(T); res['joint'].append(a['joint']); res['prod'].append(a['prod']); res['diff'].append(a['joint']-a['prod']); res['e0'].append(a['e0']); res['band'].append(a['joint_band'])
for k,v in res.items():
    v=np.array(v); print('boot',k,np.round(np.percentile(v,2.5,axis=0),3),np.round(np.percentile(v,97.5,axis=0),3))
print('diff>0 share',(np.array(res['diff'])>0).mean(),'B',B)
