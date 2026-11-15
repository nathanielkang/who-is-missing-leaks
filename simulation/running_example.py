"""Numbers of the running example (citizenship always observed; income reported w.p. 0.9 by citizens, 0.6 by non-citizens)."""
import numpy as np, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Lambda, Phi, hidden_lp, F
from scipy.optimize import linprog
g1,g0=0.6,0.9; es=ea=0.5
print('eps1',abs(np.log(g1/g0)),'eps0',abs(np.log((1-g1)/(1-g0))),'sum',abs(np.log(g1/(1-g1))-np.log(g0/(1-g0))))
# hidden: W = {(h, income observed/missing)}, neighbours differ in h only
pts=[(h,r) for h in (1,0) for r in (1,0)]
def lev(metric):
    d=np.array([[metric(u,v) for v in pts] for u in pts])
    P=np.array([g1,1-g1,0,0]); Q=np.array([0,0,g0,1-g0])
    return max(hidden_lp(P,Q,d),hidden_lp(Q,P,d))
print('record-level (eps0=0.5):',lev(lambda u,v:0.5*(u!=v)))
print('item-level LP:',lev(lambda u,v: es*(u[0]!=v[0])+ea*(u[1]!=v[1])),' closed:',max(es+Phi(g1,g0,ea),es+Phi(g0,g1,ea)),' Phi(.6,.9)=',Phi(g1,g0,ea),' Phi(.9,.6)=',Phi(g0,g1,ea),' Lambda(1,1)=',Lambda(1.0,1.0,es,ea))
print('D =',max(es,ea)+ea)
print('Thm cert(iii), rho=0.6, Gamma=1: eps0<=',np.log(1+0.6*(np.e-1)),' eps1<=',np.log(1+0.4*(np.e-1)))
# level-zero policy, one field
q0=g0*(2/3); print('citizen release prob',q0,'= non-citizen',g1)
# two independent fields: disclosed level, transport value
def pat(g): return np.array([g*g, g*(1-g), (1-g)*g, (1-g)*(1-g)])   # 11,10,01,00
k1,k0=pat(g1),pat(g0); print('two fields disclosed level',np.max(np.abs(np.log(k1/k0))), 2*np.log(4))
sz=np.array([[2,1,1,0],[1,1,0,0],[1,0,1,0],[0,0,0,0]])  # |r ^ r'|
Ae=np.zeros((8,16))
for r in range(4): Ae[r,r*4:(r+1)*4]=1; Ae[4+r,r::4]=1
res=linprog(-sz.ravel(),A_eq=Ae,b_eq=np.r_[k0,k1],bounds=[(0,None)]*16,method='highs'); print('transport fields',-res.fun,'non-citizens report',2*g1,'citizens report',2*g0)
# box
c=1.1; L=0.55; E=0.2
print('boxes contain truth:',0.85<=g0<=0.85*c, 0.55<=g1<=0.55*c, ' L_S=',min(0.85**2,0.55**2),' log C_S=',np.log(c*c))
blk=F(E,0.55**2,c*c); pr=F(E/2,L,c); print('block task',blk,' product per-field',pr,' product task',pr**2,' fields product',2*pr,' fields block',2*blk, ' one field alone',F(E,L,c))
# sample size
K=2; d=2; beta=0.05; gmin=0.55; N=np.log(4*K/beta)/(2*gmin**2*np.tanh(E/(2*d))**2); b=np.sqrt(np.log(4*K/beta)/(2*np.ceil(N)))
print('N',N,'ceil',np.ceil(N),' widths',(g1+b)/(g1-b),(g0+b)/(g0-b),' c^2<=e^E:',((g1+b)/(g1-b))**2,np.exp(E))
# selection bias
pi=0.2; sh=pi*g1/(pi*g1+(1-pi)*g0); print('share among respondents',sh,'bias',pi-sh,' tanh(1.386/4)',np.tanh(np.log(4)/4),' tanh(0.05)',np.tanh(0.05))
