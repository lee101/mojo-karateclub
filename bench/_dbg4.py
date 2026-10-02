import numpy as np
rng = np.random.default_rng(7)
d=3
a0=rng.normal(size=(d,d)); a0=np.ascontiguousarray(a0+a0.T)

def packed(a0, d=3, sweeps=100, tol=1e-24):
    a = a0.copy()
    vt = np.eye(d)
    for sweep in range(sweeps):
        off = sum(a[i,j]**2 for i in range(d) for j in range(i+1,d))
        if off <= tol: return a, vt.T, sweep
        for p in range(d):
            for q in range(p+1,d):
                apq=a[p,q]
                if apq==0.0: continue
                app=a[p,p]; aqq=a[q,q]
                theta=(aqq-app)/(2.0*apq)
                sign=1.0 if theta>=0.0 else -1.0
                t=sign/(abs(theta)+np.sqrt(theta*theta+1.0))
                c=1.0/np.sqrt(t*t+1.0); s=t*c
                # span [0,p)
                for k in range(0,p):
                    ap=a[p,k]; aq=a[q,k]
                    a[p,k]=c*ap-s*aq; a[q,k]=s*ap+c*aq
                    m=vt[p,k]; w=vt[q,k]
                    vt[p,k]=c*m-s*w; vt[q,k]=s*m+c*w
                # span (p,q)
                for k in range(p+1,q):
                    ap=a[k,p]; aq=a[q,k]
                    a[k,p]=c*ap-s*aq; a[q,k]=s*ap+c*aq
                    m=vt[p,k]; w=vt[q,k]
                    vt[p,k]=c*m-s*w; vt[q,k]=s*m+c*w
                # span (q,d)
                for k in range(q+1,d):
                    ap=a[k,p]; aq=a[k,q]
                    a[k,p]=c*ap-s*aq; a[k,q]=s*ap+c*aq
                    m=vt[p,k]; w=vt[q,k]
                    vt[p,k]=c*m-s*w; vt[q,k]=s*m+c*w
                t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
                a[p,p]=c*t1-s*t3; a[p,q]=c*t2-s*t4; a[q,q]=s*t2+c*t4
                m=vt[p,p].copy(); w=vt[q,p].copy(); vt[p,p]=c*m-s*w; vt[q,p]=s*m+c*w
                m=vt[p,q].copy(); w=vt[q,q].copy(); vt[p,q]=c*m-s*w; vt[q,q]=s*m+c*w
    return a, vt.T, sweeps

A,V,u = packed(a0,d)
print(u, np.diag(A), np.linalg.eigvalsh(a0), np.abs(a0@V-V*np.diag(A)).max())
