import numpy as np
rng = np.random.default_rng(7)
d=3
a0=rng.normal(size=(d,d)); a0=np.ascontiguousarray(a0+a0.T)
def rot_old(A,p,q,c,s):
    app=A[p,p]; aqq=A[q,q]; apq=A[p,q]
    for lo,hi in ((0,p),(p+1,q),(q+1,d)):
        x=A[lo:hi,p].copy(); y=A[lo:hi,q].copy()
        u=c*x-s*y; v=s*x+c*y
        A[lo:hi,p]=u; A[lo:hi,q]=v; A[p,lo:hi]=u; A[q,lo:hi]=v
    t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
    A[p,p]=c*t1-s*t3; A[q,p]=s*t1+c*t3
    A[p,q]=c*t2-s*t4; A[q,q]=s*t2+c*t4
def rot_packed(a,p,q,c,s):
    app=a[p,p]; aqq=a[q,q]; apq=a[p,q]
    for k in range(0,p):
        ap=a[p,k]; aq=a[q,k]; a[p,k]=c*ap-s*aq; a[q,k]=s*ap+c*aq
    for k in range(p+1,q):
        ap=a[k,p]; aq=a[q,k]; a[k,p]=c*ap-s*aq; a[q,k]=s*ap+c*aq
    for k in range(q+1,d):
        ap=a[k,p]; aq=a[k,q]; a[k,p]=c*ap-s*aq; a[k,q]=s*ap+c*aq
    t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
    a[p,p]=c*t1-s*t3; a[p,q]=c*t2-s*t4; a[q,q]=s*t2+c*t4
for p,q in [(0,1),(0,2),(1,2)]:
    A=a0.copy(); app=A[p,p];aqq=A[q,q];apq=A[p,q]
    theta=(aqq-app)/(2.0*apq); sign=1.0 if theta>=0 else -1.0
    t=sign/(abs(theta)+np.sqrt(theta*theta+1)); c=1/np.sqrt(t*t+1); s=t*c
    rot_old(A,p,q,c,s)
    a=a0.copy(); rot_packed(a,p,q,c,s)
    lo = a.copy()
    for i in range(d):
        for j in range(i+1,d): lo[j,i]=lo[i,j]
    print("pair",p,q); print("old\n",A); print("packed(mirrored)\n",lo)
