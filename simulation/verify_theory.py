import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, numpy as np, itertools
sys.path.insert(0,HERE)
from core import *; from lps import *
from scipy.optimize import linprog
rng=np.random.default_rng(7); rep={}
# (1) Corollary frame: record-level metric, G* = max_J (1+cP(J))/(1+cQ(J)), fixed point, K=1 formula
mx=0; mxfp=0
for _ in range(30):
    n=6; P=rng.dirichlet(np.ones(n)); Q=rng.dirichlet(np.ones(n)); e0=rng.uniform(0.2,2); c=np.exp(e0)-1
    if rng.random()<0.5: P[rng.integers(n)]=0; P/=P.sum()
    d=e0*(1-np.eye(n)); lp=np.exp(hidden_lp(P,Q,d))
    best=max((1+c*P[list(J)].sum())/(1+c*Q[list(J)].sum()) for m in range(n+1) for J in itertools.combinations(range(n),m))
    mx=max(mx,abs(lp-best)); HS=np.maximum(P-best*Q,0).sum(); mxfp=max(mxfp,abs(best-1-c*HS))
rep['cor_frame_LP_vs_formula']=mx; rep['cor_frame_fixed_point']=mxfp
mx=0
for _ in range(30):
    g,gp=rng.random(2); e0=rng.uniform(0.2,2); c=np.exp(e0)-1
    P=np.array([g,0,1-g]); Q=np.array([0,gp,1-gp]); d=e0*(1-np.eye(3))
    two=max(hidden_lp(P,Q,d),hidden_lp(Q,P,d)); mx=max(mx,abs(two-np.log(1+max(g,gp)*c)))
rep['cor_frame_K1']=mx
# (2) Theorem closed: one changed item with partial response + unchanged items -> Lambda + sum Phi  (generic check with changed optional item)
mx=0
for _ in range(30):
    ea=rng.uniform(0.2,1.5); es=rng.uniform(0,2*ea); g,gp=rng.uniform(0.05,0.95,2); g2,g2p=rng.uniform(0.05,0.95,2)
    # W: item1 in {y, y', bot}, item2 in {v, bot}
    pts=[(a,b) for a in ('y','yp','bot') for b in ('v','bot')]
    def dist(u,v):
        s=0
        if u[0]!=v[0]: s+= es if 'bot' not in (u[0],v[0]) else ea
        if u[1]!=v[1]: s+= ea
        return s
    d=np.array([[dist(u,v) for v in pts] for u in pts])
    P=np.array([ (g if a=='y' else (1-g if a=='bot' else 0))*(g2 if b=='v' else 1-g2) for a,b in pts])
    Q=np.array([ (gp if a=='yp' else (1-gp if a=='bot' else 0))*(g2p if b=='v' else 1-g2p) for a,b in pts])
    mx=max(mx,abs(hidden_lp(P,Q,d)-(Lambda(g,gp,es,ea)+Phi(g2,g2p,ea))))
rep['thm_closed_changed_plus_unchanged']=mx
# (3) Proposition worst: sup over kappa of log G* = D (point masses), item metric
ea,es=0.7,0.5
rep['prop_worst_D_formula']=float(1*max(es,ea)+4*ea)
# (4) Theorem transport: LP value == 0.5(E0 w + E1 w - W) and independent case
mx=0; mx2=0
for _ in range(20):
    k0=rng.dirichlet(np.ones(16)); k1=rng.dirichlet(np.ones(16)); w=rng.uniform(0.2,2,4)
    v,q,pi=transport_zero(k0,k1,w)
    cost=np.array([[w@np.abs(PAT[r]-PAT[rp]) for rp in range(16)] for r in range(16)]).ravel()
    Ae=np.zeros((32,256))
    for r in range(16): Ae[r,r*16:(r+1)*16]=1; Ae[16+r,r::16]=1
    W=linprog(cost,A_eq=Ae,b_eq=np.r_[k0,k1],bounds=[(0,None)]*256,method='highs').fun
    mx=max(mx,abs(v-0.5*(w@margins(k0)+w@margins(k1)-W)))
    g0,g1=rng.random(4),rng.random(4); v2,_,_=transport_zero(prod_from_margins(g0),prod_from_margins(g1),w); mx2=max(mx2,abs(v2-w@np.minimum(g0,g1)))
rep['thm_transport_identity']=mx; rep['thm_transport_indep']=mx2
k0=np.zeros(16);k1=np.zeros(16)
# two-field example embedded in fields 0,1
k0[0]=.5;k0[3]=.5;k1[1]=.5;k1[2]=.5
rep['transport_example_half']=transport_zero(k0,k1,np.array([1,1,0,0.]))[0]
# (5) Theorem joint (ii): max tau_S over faithful policies of level E on a common-width box equals F(E,L_S,C_S)
mx=0
for _ in range(12):
    ck=rng.uniform(1.0,1.25,4); ell=np.array([rng.uniform(0.4,0.75,4),rng.uniform(0.4,0.75,4)]); ell=np.minimum(ell,1/ck*0.999)
    lo=ell; hi=ell*ck; E=rng.uniform(0.05,1.2)
    Q0,Q1,tau,tS=exact_product_box(lo,hi,E,'task')
    mx=max(mx,abs(tS-F(E,ell.prod(1).min(),ck.prod())))
rep['thm_joint_ii_block_optimal']=mx
# (6) Theorem joint (iii): balanced product policy has level sum phi over the box; exact fields >= balanced
mx=0; mn=1
for _ in range(12):
    ck=rng.uniform(1.02,1.2,4); ell=np.array([rng.uniform(0.4,0.75,4),rng.uniform(0.4,0.75,4)]); E=rng.uniform(0.2,1.2)
    tau=balanced_alloc(E,ck,ell.min(0)); pols=(product_policy(tau/ell[0]),product_policy(tau/ell[1]))
    lev=0
    for v0 in range(16):
        g0=np.where([(v0>>k)&1 for k in range(4)],ell[0]*ck,ell[0]); q0=released(prod_from_margins(g0),pols[0])
        for v1 in range(16):
            g1=np.where([(v1>>k)&1 for k in range(4)],ell[1]*ck,ell[1]); q1=released(prod_from_margins(g1),pols[1]); lev=max(lev,level(q0,q1))
    mx=max(mx,abs(lev-sum(phi_c(c,t) for c,t in zip(ck,tau))))
    _,_,tex,_=exact_product_box(ell,ell*ck,E,'fields'); mn=min(mn,tex.sum()-tau.sum())
rep['thm_joint_iii_balanced_level']=mx; rep['exact_minus_balanced_min']=mn
# (7) Theorem ign (i): sup over two-group populations of |E[f|T=t]-E f| = tanh(E/4)
best=0; E=0.9; c=np.exp(E)
for nu in np.linspace(0.001,0.999,999):
    best=max(best,nu*(1-nu)*(c-1)/(1+nu*(c-1)))
rep['thm_ign_sup']=best; rep['tanh(E/4)']=np.tanh(E/4)
# (8) examples in the text
rep['covariate_cells_level']=max(abs(np.log(.95/.75)),abs(np.log(.05/.25))); rep['log5']=np.log(5)
rep['joint_example']=float(np.max(np.abs(np.log(np.array([.45,.05,.05,.45])/np.array([.05,.45,.45,.05]))))); rep['log9']=np.log(9)
# sample size
for E in (0.5,1.0): rep[f'N_E{E}']=np.log(16/(0.05/3))/(2*0.8**2*np.tanh(E/8)**2)
for k,v in rep.items(): print(k, v)
