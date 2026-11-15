import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, numpy as np, json
sys.path.insert(0,HERE)
from core import *; from lps import transport_zero
out={}
# NHANES: zero-level transport at the true (Jeffreys) law, within age bands
C=np.load(HERE + '/data/nhanes_counts.npy'); tot=C.sum(0); N=tot.sum(2); wb=N.sum(1)/N.sum()
kap=np.array([[jeffreys(tot[b,h]) for h in range(2)] for b in range(3)])
f=0;t=0;obs=0;wh=0;nsf=0;nst=0;levs=[]
for b in range(3):
    v,q,pi=transport_zero(kap[b,0],kap[b,1]); g=np.array([margins(kap[b,h]) for h in range(2)])
    f+=wb[b]*v; t+=wb[b]*q[15]; nsf+=wb[b]*g.min(0).sum(); nst+=wb[b]*min(kap[b,0,15],kap[b,1,15])
    o=(N[b,0]*g[0].sum()+N[b,1]*g[1].sum())/N[b].sum(); obs+=wb[b]*o; wh+=wb[b]*(o-v); levs.append(level(kap[b,0],kap[b,1]))
out['nhanes']=dict(transport_fields=f,transport_task=t,nosupp_fields=nsf,nosupp_task=nst,observed_per_adult=obs,withheld_per_adult=wh,band_levels=levs,
   dev_nosupp=[float(abs((N[b,1]/N[b].sum())*kap[b,1,15]/((N[b,1]/N[b].sum())*kap[b,1,15]+(N[b,0]/N[b].sum())*kap[b,0,15])-N[b,1]/N[b].sum())) for b in range(3)])
# ACS extract (reported citizenship)
S=np.load(HERE + '/data/acs_extract_reported.npy'); n=S.sum(2); wb=n.sum(1)/n.sum(); Pb=S/S.sum(2,keepdims=True)
f=0;obs=0;wh=0;nsf=0;t=0
for b in range(3):
    v,q,pi=transport_zero(Pb[b,0],Pb[b,1]); g=np.array([margins(Pb[b,h]) for h in range(2)])
    o=(n[b,0]*g[0].sum()+n[b,1]*g[1].sum())/n[b].sum(); f+=wb[b]*v; obs+=wb[b]*o; wh+=wb[b]*(o-v); nsf+=wb[b]*g.min(0).sum(); t+=wb[b]*q[15]
pooled=S.sum(0); P=pooled/pooled.sum(1,keepdims=True); vp,qp,_=transport_zero(P[0],P[1])
out['acs']=dict(transport_fields_band=f,transport_task_band=t,nosupp_fields_band=nsf,observed_per_adult=obs,withheld_per_adult_band=wh,transport_fields_pooled=vp,
   observed_pooled=float((pooled.sum(0)@SIZE)/pooled.sum()), withheld_pooled=float((pooled.sum(0)@SIZE)/pooled.sum()-vp), D=0.5+4*0.5)
print(json.dumps(out,indent=1,default=float)); json.dump(out,open(HERE + '/results/out_extra.json','w'),indent=1,default=float)
