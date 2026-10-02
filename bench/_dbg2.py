import numpy as np
def new_jacobi(a, d=3, sweeps=100, tol=1e-24):
    vectors = a.T.copy()
    vt = np.eye(d)
    used = sweeps
    for sweep in range(sweeps):
        off = 0.0
        for i in range(d):
            for j in range(i+1, d):
                off += vectors[i, j]**2
        if off <= tol:
            used = sweep; break
        for p in range(d):
            for q in range(p+1, d):
                apq = vectors[q, p]
                if apq == 0.0: continue
                app = vectors[p, p]; aqq = vectors[q, q]
                theta = (aqq - app) / (2.0 * apq)
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + np.sqrt(theta*theta+1.0))
                c = 1.0/np.sqrt(t*t+1.0); s = t*c
                for lo,hi in ((0,p),(p+1,q),(q+1,d)):
                    ap = vectors[p, lo:hi].copy(); aq = vectors[q, lo:hi].copy()
                    bp = vt[p, lo:hi].copy(); bq = vt[q, lo:hi].copy()
                    vectors[p, lo:hi] = c*ap - s*aq
                    vectors[q, lo:hi] = s*ap + c*aq
                    vt[p, lo:hi] = c*bp - s*bq
                    vt[q, lo:hi] = s*bp + c*bq
                t1 = c*app - s*apq; t2 = s*app + c*apq
                t3 = c*apq - s*aqq; t4 = s*apq + c*aqq
                vectors[p,p] = c*t1 - s*t3
                vectors[p,q] = s*t1 + c*t3
                vectors[q,p] = c*t2 - s*t4
                vectors[q,q] = s*t2 + c*t4
                m = vt[p,p].copy(); w = vt[q,p].copy()
                vt[p,p] = c*m - s*w; vt[q,p] = s*m + c*w
                m = vt[p,q].copy(); w = vt[q,q].copy()
                vt[p,q] = c*m - s*w; vt[q,q] = s*m + c*w
    A = vectors.T.copy()
    V = vt.T.copy()
    return A, V, used

rng = np.random.default_rng(7)
d=3
a=rng.normal(size=(d,d)); a=np.ascontiguousarray(a+a.T)
A,V,used = new_jacobi(a,d)
print("used",used); print(np.diag(A)); print(np.linalg.eigvalsh(a)); print(np.abs(a@V-V*np.diag(A)).max())

def old_jacobi(a, d=3, sweeps=100, tol=1e-24):
    A = a.copy(); vt = np.eye(d); used=sweeps
    for sweep in range(sweeps):
        off = sum(A[i,j]**2 for i in range(d) for j in range(i+1,d))
        if off <= tol: used=sweep; break
        for p in range(d):
            for q in range(p+1,d):
                apq = A[p,q]
                if apq==0.0: continue
                app=A[p,p]; aqq=A[q,q]
                theta=(aqq-app)/(2.0*apq)
                sign=1.0 if theta>=0.0 else -1.0
                t=sign/(abs(theta)+np.sqrt(theta*theta+1.0))
                c=1.0/np.sqrt(t*t+1.0); s=t*c
                for lo,hi in ((0,p),(p+1,q),(q+1,d)):
                    x = A[lo:hi,p].copy(); y=A[lo:hi,q].copy()
                    u=c*x-s*y; v=s*x+c*y
                    A[lo:hi,p]=u; A[lo:hi,q]=v
                    A[p,lo:hi]=u; A[q,lo:hi]=v
                    m=vt[p,lo:hi].copy(); w=vt[q,lo:hi].copy()
                    vt[p,lo:hi]=c*m-s*w; vt[q,lo:hi]=s*m+c*w
                t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
                A[p,p]=c*t1-s*t3; A[q,p]=s*t1+c*t3
                A[p,q]=c*t2-s*t4; A[q,q]=s*t2+c*t4
                m=vt[p,p].copy(); w=vt[q,p].copy(); vt[p,p]=c*m-s*w; vt[q,p]=s*m+c*w
                m=vt[p,q].copy(); w=vt[q,q].copy(); vt[p,q]=c*m-s*w; vt[q,q]=s*m+c*w
    return A, vt.T.copy(), used

A2,V2,u2 = old_jacobi(a,d)
print("old used",u2); print(np.diag(A2)); print(np.abs(a@V2-V2*np.diag(A2)).max())
