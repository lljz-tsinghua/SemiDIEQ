import numpy as np
from scipy import optimize
d=3
k=2
rk=2
niter=1000
eps=1e-8
regbig=1e-5
regsmall=1e-8
penalty=1e2
def fun(x):
    a=np.array(x).reshape(-1,d,d)
    MatE0=a[0*d:1*d,:,:]+1j*a[1*d:2*d,:,:]
    MatE1=a[2*d:3*d,:,:]+1j*a[3*d:4*d,:,:]
    MatF0=a[4*d:5*d,:,:]+1j*a[5*d:6*d,:,:]
    MatF1=a[6*d:7*d,:,:]+1j*a[7*d:8*d,:,:]
    E0=np.zeros_like(MatE0).astype(np.complex128)
    E1=np.zeros_like(MatE1).astype(np.complex128)
    F0=np.zeros_like(MatF0).astype(np.complex128)
    F1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        E0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        E1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        F0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        F1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])
    sum_E0=np.sum(E0,axis=(0))
    sum_E1=np.sum(E1,axis=(0))
    sum_F0=np.sum(F0,axis=(0))
    sum_F1=np.sum(F1,axis=(0))

    try:
        normalizer_E0=np.linalg.inv(np.linalg.cholesky(sum_E0).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_E0 is not positive definite!')
        print(sum_E0)
        return 1e10000

    try:
        normalizer_E1=np.linalg.inv(np.linalg.cholesky(sum_E1).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_E1 is not positive definite!')
        print(sum_E1)
        return 1e10000

    try:
        normalizer_F0=np.linalg.inv(np.linalg.cholesky(sum_F0).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_F0 is not positive definite!')
        print(sum_F0)
        return 1e10000

    try:
        normalizer_F1=np.linalg.inv(np.linalg.cholesky(sum_F1).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_F1 is not positive definite!')
        print(sum_F1)
        return 1e10000

    for i in range(d):
        MatE0[i]=np.dot(MatE0[i],normalizer_E0)
        MatE1[i]=np.dot(MatE1[i],normalizer_E1)
        MatF0[i]=np.dot(MatF0[i],normalizer_F0)
        MatF1[i]=np.dot(MatF1[i],normalizer_F1)
    M0=np.zeros_like(MatE0).astype(np.complex128)
    M1=np.zeros_like(MatE1).astype(np.complex128)
    N0=np.zeros_like(MatF0).astype(np.complex128)
    N1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        M0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        M1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        N0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        N1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])
    A=np.zeros((d*d,d*d)).astype(np.complex128)
    for i in range(d):
        for j in range(i+1):
            A+=(M1[i][:,None,:,None]*N1[j][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[j][:,None,:,None]*N1[i][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[i][:,None,:,None]*N0[j][None,:,None,:]).reshape(d*d,d*d)
            if j<i:
                A+=(M1[j][:,None,:,None]*N0[i][None,:,None,:]).reshape(d*d,d*d)
    eigenlist=np.sort(np.real(np.linalg.eigvals(A)))
    #print(eigenlist)
    #return -(eigenlist[-1])
    return -np.sum(eigenlist[::-1][:k])

def extremify(x):
    dist=1e5
    for i in range(d):
        eigenlist=np.real(np.linalg.eigvals(x[i]))
        for eig in eigenlist:
            if 1.0/d>eig+eps:
                dist=min(dist,eig/(1.0/d-eig))
    mix=np.repeat((np.identity(d)/d)[None,:,:],d,axis=(0))
    return (x+dist*(x-mix))

def funb(x):
    a=np.array(x).reshape(-1,d,d)
    MatE0=a[0*d:1*d,:,:]+1j*a[1*d:2*d,:,:]
    MatE1=a[2*d:3*d,:,:]+1j*a[3*d:4*d,:,:]
    MatF0=a[4*d:5*d,:,:]+1j*a[5*d:6*d,:,:]
    MatF1=a[6*d:7*d,:,:]+1j*a[7*d:8*d,:,:]
    E0=np.zeros_like(MatE0).astype(np.complex128)
    E1=np.zeros_like(MatE1).astype(np.complex128)
    F0=np.zeros_like(MatF0).astype(np.complex128)
    F1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        E0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        E1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        F0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        F1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])
    sum_E0=np.sum(E0,axis=(0))
    sum_E1=np.sum(E1,axis=(0))
    sum_F0=np.sum(F0,axis=(0))
    sum_F1=np.sum(F1,axis=(0))

    try:
        normalizer_E0=np.linalg.inv(np.linalg.cholesky(sum_E0).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_E0 is not positive definite!')
        print(sum_E0)
        return 1e10000

    try:
        normalizer_E1=np.linalg.inv(np.linalg.cholesky(sum_E1).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_E1 is not positive definite!')
        print(sum_E1)
        return 1e10000

    try:
        normalizer_F0=np.linalg.inv(np.linalg.cholesky(sum_F0).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_F0 is not positive definite!')
        print(sum_F0)
        return 1e10000

    try:
        normalizer_F1=np.linalg.inv(np.linalg.cholesky(sum_F1).transpose().conjugate())
    except np.linalg.LinAlgError:
        print('sum_F1 is not positive definite!')
        print(sum_F1)
        return 1e10000

    for i in range(d):
        MatE0[i]=np.dot(MatE0[i],normalizer_E0)
        MatE1[i]=np.dot(MatE1[i],normalizer_E1)
        MatF0[i]=np.dot(MatF0[i],normalizer_F0)
        MatF1[i]=np.dot(MatF1[i],normalizer_F1)
    M0=np.zeros_like(MatE0).astype(np.complex128)
    M1=np.zeros_like(MatE1).astype(np.complex128)
    N0=np.zeros_like(MatF0).astype(np.complex128)
    N1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        M0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        M1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        N0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        N1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])

    # Extremify the POVM
    M0=extremify(M0)
    M1=extremify(M1)
    N0=extremify(N0)
    N1=extremify(N1)

    A=np.zeros((d*d,d*d)).astype(np.complex128)
    for i in range(d):
        for j in range(i+1):
            A+=(M1[i][:,None,:,None]*N1[j][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[j][:,None,:,None]*N1[i][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[i][:,None,:,None]*N0[j][None,:,None,:]).reshape(d*d,d*d)
            if j<i:
                A+=(M1[j][:,None,:,None]*N0[i][None,:,None,:]).reshape(d*d,d*d)
    eigenlist=np.sort(np.real(np.linalg.eigvals(A)))
    #return -(eigenlist[-1])
    return -np.sum(eigenlist[::-1][:k])

def fun2(x):
    a=np.array(x).reshape(-1,d,d)
    MatA0=a[0,:,:]+1j*a[1,:,:]
    MatA1=a[2,:,:]+1j*a[3,:,:]
    MatB0=a[4,:,:]+1j*a[5,:,:]
    MatB1=a[6,:,:]+1j*a[7,:,:]
    BasA0=np.linalg.qr(MatA0)[0]
    BasA1=np.linalg.qr(MatA1)[0]
    BasB0=np.linalg.qr(MatB0)[0]
    BasB1=np.linalg.qr(MatB1)[0]
    b=np.zeros((8*d,d,d)).astype(np.complex128)
    for i in range(d):
        b[0*d+i,:,:]=np.dot(BasA0[:,i][:,None],BasA0[i,:][None,:]).real
        b[1*d+i,:,:]=np.dot(BasA0[:,i][:,None],BasA0[i,:][None,:]).imag
        b[2*d+i,:,:]=np.dot(BasA1[:,i][:,None],BasA1[i,:][None,:]).real
        b[3*d+i,:,:]=np.dot(BasA1[:,i][:,None],BasA1[i,:][None,:]).imag
        b[4*d+i,:,:]=np.dot(BasB0[:,i][:,None],BasB0[i,:][None,:]).real
        b[5*d+i,:,:]=np.dot(BasB0[:,i][:,None],BasB0[i,:][None,:]).imag
        b[6*d+i,:,:]=np.dot(BasB1[:,i][:,None],BasB1[i,:][None,:]).real
        b[7*d+i,:,:]=np.dot(BasB1[:,i][:,None],BasB1[i,:][None,:]).imag
    return fun(b)

def fun3(x):
    a=np.array(x).reshape(-1,d,d)
    MatA0=a[0,:,:]+1j*a[1,:,:]
    MatA1=a[2,:,:]+1j*a[3,:,:]
    MatB0=a[4,:,:]+1j*a[5,:,:]
    MatB1=a[6,:,:]+1j*a[7,:,:]
    b=np.zeros((8*d,d,d)).astype(np.complex128)
    for i in range(d):
        b[0*d+i,:,:]=np.dot(MatA0[:,i][:,None].conjugate(),MatA0[i,:][None,:]).real
        b[1*d+i,:,:]=np.dot(MatA0[:,i][:,None].conjugate(),MatA0[i,:][None,:]).imag
        b[2*d+i,:,:]=np.dot(MatA1[:,i][:,None].conjugate(),MatA1[i,:][None,:]).real
        b[3*d+i,:,:]=np.dot(MatA1[:,i][:,None].conjugate(),MatA1[i,:][None,:]).imag
        b[4*d+i,:,:]=np.dot(MatB0[:,i][:,None].conjugate(),MatB0[i,:][None,:]).real
        b[5*d+i,:,:]=np.dot(MatB0[:,i][:,None].conjugate(),MatB0[i,:][None,:]).imag
        b[6*d+i,:,:]=np.dot(MatB1[:,i][:,None].conjugate(),MatB1[i,:][None,:]).real
        b[7*d+i,:,:]=np.dot(MatB1[:,i][:,None].conjugate(),MatB1[i,:][None,:]).imag
    return fun(b)

def fun4(x):
    a=np.array(x).reshape(-1,rk,d,d)
    MatA0=a[0,:,:,:]+1j*a[1,:,:,:]
    MatA1=a[2,:,:,:]+1j*a[3,:,:,:]
    MatB0=a[4,:,:,:]+1j*a[5,:,:,:]
    MatB1=a[6,:,:,:]+1j*a[7,:,:,:]
    b=np.zeros((8*d,d,d)).astype(np.complex128)
    for j in range(rk):
        for i in range(d):
            par=np.random.uniform(0,1,(4))
            b[0*d+i,:,:]+=par[0]*np.dot(MatA0[j,:,i][:,None],MatA0[j,i,:][None,:]).real
            b[1*d+i,:,:]+=par[0]*np.dot(MatA0[j,:,i][:,None],MatA0[j,i,:][None,:]).imag
            b[2*d+i,:,:]+=par[1]*np.dot(MatA1[j,:,i][:,None],MatA1[j,i,:][None,:]).real
            b[3*d+i,:,:]+=par[1]*np.dot(MatA1[j,:,i][:,None],MatA1[j,i,:][None,:]).imag
            b[4*d+i,:,:]+=par[2]*np.dot(MatB0[j,:,i][:,None],MatB0[j,i,:][None,:]).real
            b[5*d+i,:,:]+=par[2]*np.dot(MatB0[j,:,i][:,None],MatB0[j,i,:][None,:]).imag
            b[6*d+i,:,:]+=par[3]*np.dot(MatB1[j,:,i][:,None],MatB1[j,i,:][None,:]).real
            b[7*d+i,:,:]+=par[3]*np.dot(MatB1[j,:,i][:,None],MatB1[j,i,:][None,:]).imag
    return fun(b)

def fun4b(x):
    a=np.array(x).reshape(-1,rk,d,d)
    MatA0=a[0,:,:,:]+1j*a[1,:,:,:]
    MatA1=a[2,:,:,:]+1j*a[3,:,:,:]
    MatB0=a[4,:,:,:]+1j*a[5,:,:,:]
    MatB1=a[6,:,:,:]+1j*a[7,:,:,:]
    b=np.zeros((8*d,d,d)).astype(np.complex128)
    for j in range(rk):
        for i in range(d):
            b[0*d+i,:,:]+=np.dot(MatA0[j,:,i][:,None],MatA0[j,i,:][None,:]).real
            b[1*d+i,:,:]+=np.dot(MatA0[j,:,i][:,None],MatA0[j,i,:][None,:]).imag
            b[2*d+i,:,:]+=np.dot(MatA1[j,:,i][:,None],MatA1[j,i,:][None,:]).real
            b[3*d+i,:,:]+=np.dot(MatA1[j,:,i][:,None],MatA1[j,i,:][None,:]).imag
            b[4*d+i,:,:]+=np.dot(MatB0[j,:,i][:,None],MatB0[j,i,:][None,:]).real
            b[5*d+i,:,:]+=np.dot(MatB0[j,:,i][:,None],MatB0[j,i,:][None,:]).imag
            b[6*d+i,:,:]+=np.dot(MatB1[j,:,i][:,None],MatB1[j,i,:][None,:]).real
            b[7*d+i,:,:]+=np.dot(MatB1[j,:,i][:,None],MatB1[j,i,:][None,:]).imag
    return b

def fun5(x):
    return funb(x)+regbig*np.sum(x**2)+regsmall*np.sum(1.0/(x**2))

def fun6(x):
    a=np.array(x).reshape(-1,d,d)
    MatE0=a[0*d:1*d,:,:]+1j*a[1*d:2*d,:,:]
    MatE1=a[2*d:3*d,:,:]+1j*a[3*d:4*d,:,:]
    MatF0=a[4*d:5*d,:,:]+1j*a[5*d:6*d,:,:]
    MatF1=a[6*d:7*d,:,:]+1j*a[7*d:8*d,:,:]
    E0=np.zeros_like(MatE0).astype(np.complex128)
    E1=np.zeros_like(MatE1).astype(np.complex128)
    F0=np.zeros_like(MatF0).astype(np.complex128)
    F1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        E0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        E1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        F0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        F1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])
    sum_E0=np.sum(E0,axis=(0))
    sum_E1=np.sum(E1,axis=(0))
    sum_F0=np.sum(F0,axis=(0))
    sum_F1=np.sum(F1,axis=(0))
    ret=0.0
    ret+=np.sum((sum_E0-np.identity(d)).real**4)
    ret+=np.sum((sum_E1-np.identity(d)).real**4)
    ret+=np.sum((sum_F0-np.identity(d)).real**4)
    ret+=np.sum((sum_F1-np.identity(d)).real**4)
    ret*=penalty
    M0=np.zeros_like(MatE0).astype(np.complex128)
    M1=np.zeros_like(MatE1).astype(np.complex128)
    N0=np.zeros_like(MatF0).astype(np.complex128)
    N1=np.zeros_like(MatF1).astype(np.complex128)
    for i in range(d):
        M0[i]=np.dot(MatE0[i].transpose().conjugate(),MatE0[i])
        M1[i]=np.dot(MatE1[i].transpose().conjugate(),MatE1[i])
        N0[i]=np.dot(MatF0[i].transpose().conjugate(),MatF0[i])
        N1[i]=np.dot(MatF1[i].transpose().conjugate(),MatF1[i])
    A=np.zeros((d*d,d*d)).astype(np.complex128)
    for i in range(d):
        for j in range(i+1):
            A+=(M1[i][:,None,:,None]*N1[j][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[j][:,None,:,None]*N1[i][None,:,None,:]).reshape(d*d,d*d)
            A+=(M0[i][:,None,:,None]*N0[j][None,:,None,:]).reshape(d*d,d*d)
            if j<i:
                A+=(M1[j][:,None,:,None]*N0[i][None,:,None,:]).reshape(d*d,d*d)
    eigenlist=np.sort(np.real(np.linalg.eigvals(A)))
    #return -(eigenlist[-1])
    return -np.sum(eigenlist[::-1][:k])+ret

global nit
nit=0
def dump(xk):
    global nit
    nit+=1
    if nit%100==0:
        print(nit,flush=True)
        print(-fun(xk))
def dump3(xk):
    global nit
    nit+=1
    if nit%100==0:
        print(nit,flush=True)
        print(-fun3(xk))

#N=8*d*d
#for i in range(niter):
#    nit=0
#    print("Iteration {}:".format(i),flush=True)
#    x0=np.random.normal(0,1,N)
#    #ans=optimize.minimize(fun3,x0,method="Nelder-Mead",callback=dump3,options={'adaptive':True,'maxfev':N*100000,'maxiter':N*100000,'fatol':1e-7})
#    ans=optimize.minimize(fun3,x0,method="L-BFGS-B",callback=dump3,options={'maxfun':N*100000,'maxiter':N*100000,'ftol':1e-7})
#    #print(ans,flush=True)
#    #print(ans.x,flush=True)
#    print(-fun3(ans.x),flush=True)

N=8*d*d*d
for i in range(niter):
    nit=0
    print("Iteration {}:".format(i),flush=True)
    x0=np.random.normal(0,1,N)
    #ans=optimize.minimize(fun,x0,method="Nelder-Mead",callback=dump,options={'adaptive':True,'maxfev':N*100000,'maxiter':N*100000,'fatol':1e-7})
    ans=optimize.minimize(fun,x0,method="L-BFGS-B",callback=dump,options={'maxfun':N*100000,'maxiter':N*100000,'ftol':1e-7})
    #print(ans,flush=True)
    #print(ans.x,flush=True)
    print(-fun(ans.x),flush=True)

