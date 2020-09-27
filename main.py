import numpy as np
import os
import math
import itertools

eps=1e-8
niter=10

def normalize_POVM(M):
    M_sum=np.sum(M,axis=(0))
    try:
        M_normalizer=np.linalg.inv(np.linalg.cholesky(M_sum))
    except np.linalg.LinAlgError:
        print('trying to normalize a matrix that is not positive definite')
        return None
    return M_normalizer@M@(M_normalizer.transpose().conjugate())

def POVM_operator_to_element(E):
    return E@(E.conjugate().transpose((0,2,1)))

def sanitize_POVM(x):
    M=x[0,:,:,:,:,:]+1.0j*x[1,:,:,:,:,:]
    n,m,k,d,_=M.shape
    for i in range(n):
        for j in range(m):
            M[i,j]=normalize_POVM(POVM_operator_to_element(M[i,j]))
    return M,n,m,k,d

def iterate_correlation(M):
    n,m,k,d,_=M.shape
    for tx in range(m):
        for ty in range(m):
            for ta in range(k):
                for tb in range(k):
                    yield (tx,ty),(ta,tb),np.kron(M[0,tx,ta],M[1,ty,tb])

def generate_correlation(x,state):
    M,n,m,k,d=sanitize_POVM(x)
    corr_shape=tuple([m]*n+[k]*n)
    corr=np.zeros(corr_shape)
    for tx,ta,tO in iterate_correlation(M):
        corr[tuple(list(tx)+list(ta))]=np.sum(state.conjugate()*tO).real
    return corr

def generate_Bell_operator(x,bf,engine=None):
    M,n,m,k,d=sanitize_POVM(x)
    if engine!=None:
        return engine(M,bf)
    BO=np.zeros((d**n,d**n)).astype(np.complex128)
    for tx,ta,tO in iterate_correlation(M):
        BO+=bf[tuple(list(tx)+list(ta))]*tO
    return BO

n,m,k,d=2,2,3,3
bf=np.loadtxt('corr_functional').reshape((2,2,k,k))
optx=np.zeros((2,2,m,k,d,d))
MA=np.zeros((m,k,d,d)).astype(np.complex128)
MB=np.zeros((m,k,d,d)).astype(np.complex128)
for s in range(m):
    for t1 in range(d):
        vecA=np.zeros((d,1)).astype(np.complex128)
        vecB=np.zeros((d,1)).astype(np.complex128)
        pA=s/m
        pB=(0.5-s)/m
        for t2 in range(d):
            vecA[t2,0]=np.exp(1.0j*2.0*math.pi/d*t2*(t1+pA))
            vecB[t2,0]=np.exp(1.0j*2.0*math.pi/d*t2*(-t1+pB))
        vecA/=np.sum(vecA*(vecA.conjugate()))**0.5
        vecB/=np.sum(vecB*(vecB.conjugate()))**0.5
        MA[s,t1]=vecA@(vecA.transpose().conjugate())
        MB[s,t1]=vecB@(vecB.transpose().conjugate())
optx[0,0,:,:,:,:]=MA.real
optx[1,0,:,:,:,:]=MA.imag
optx[0,1,:,:,:,:]=MB.real
optx[1,1,:,:,:,:]=MB.imag

BO=generate_Bell_operator(optx,bf)
v,w=np.linalg.eig(BO)
ind=np.argmax(v.real)
base_state=w[:,ind].reshape((-1,1))

for eps_state in np.logspace(-1,-3,niter):
    tmp_state=base_state+np.random.normal(0,eps_state,base_state.shape)+1.0j*np.random.normal(0,eps_state,base_state.shape)
    tmp_state/=np.sum(tmp_state*(tmp_state.conjugate()))**0.5
    tmp_state=tmp_state.reshape(-1,1)@(tmp_state.reshape(1,-1).conjugate())
    corr=generate_correlation(optx,tmp_state)
    np.savetxt('corr',corr.reshape(-1))
    os.system('./bound')

