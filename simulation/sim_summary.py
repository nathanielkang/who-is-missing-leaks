import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, json, numpy as np
res=[]
for f in sys.argv[1:]: res+=json.load(open(f))
R=len(res); print('reps',R)
POL=['none','fitted','balprod','block','exprod','exjoint_f','exjoint_t']; ES=(0.5,1.0)
S={'reps':R}
for mult in ('1','4'):
    covj=np.array([r[mult]['cov_j'] for r in res]); covp=np.array([r[mult]['cov_p'] for r in res]); hoef=np.array([r[mult]['hoef'] for r in res])
    S[f'cov_j|{mult}']=float(covj.mean()); S[f'cov_p|{mult}']=float(covp.mean()); S[f'hoef|{mult}']=hoef.mean(0).tolist()
    print('mult',mult,'cov_joint',covj.mean(),'cov_prod',covp.mean(),'hoef',hoef.mean(0))
    for ei,E in enumerate(ES):
        for p in POL:
            a=np.array([[np.nan if x is None else x for x in r[mult][f'{p}|{ei}']] for r in res],float)
            ov=a[:,0]>E+1e-4; over=ov.mean()*100
            cov = covj if p.startswith('exjoint') else covp
            S[f'{mult}|{E}|{p}']=dict(over=float(over),fields=float(a[:,1].mean()),task=float(a[:,2].mean()),dev=float(np.nanmax(a[:,3])),maxlev=float(a[:,0].max()),
                 n_over=int(ov.sum()), over_and_covered=int((ov&cov).sum()), se_f=float(a[:,1].std()/np.sqrt(R)), se_t=float(a[:,2].std()/np.sqrt(R)))
            print(f'  E={E} {p:10s} over {over:6.2f} (n={ov.sum()}, covered&over={(ov&cov).sum()}) fields {a[:,1].mean():.3f} task {a[:,2].mean():.3f} maxdev {np.nanmax(a[:,3]):.4f} maxlev {a[:,0].max():.3f}')
    for ei,E in enumerate(ES):
        for b in range(3):
            c=np.array([[np.nan if x is None else x for x in r[mult]['cc'][f'{b}|{ei}']] for r in res],float)
            rel=np.isfinite(c[:,0])
            print(f'  CC E={E} band {b} rmse {np.sqrt(np.nanmean(c[:,0]**2)):.4f} released {rel.mean():.4f} minNS {c[:,1].min()} m0 range {c[:,2].min()}-{c[:,2].max()} tauS mean {c[:,3].mean():.3f}')
            S[f'cc|{mult}|{E}|{b}']=dict(rmse=float(np.sqrt(np.nanmean(c[:,0]**2))),released=float(rel.mean()),bias=float(np.nanmean(c[:,0])))
json.dump(S,open(HERE + '/results/out_sim_summary.json','w'),indent=1)
