import numpy as np
rng = np.random.default_rng(7)
d=3
a0=rng.normal(size=(d,d)); a0=np.ascontiguousarray(a0+a0.T)
print(a0)
def rot_old(A, p,q,c,s):
    app=A[p,p]; aqq=A[q,q]; apq=A[p,q]
    for lo,hi in ((0,p),(p+1,q),(q+1,d)):
        x=A[lo:hi,p].copy(); y=A[lo:hi,q].copy()
        u=c*x-s*y; v=s*x+c*y
        A[lo:hi,p]=u; A[lo:hi,q]=v; A[p,lo:hi]=u; A[q,lo:hi]=v
    t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
    A[p,p]=c*t1-s*t3; A[q,p]=s*t1+c*t3
    A[p,q]=c*t2-s*t4; A[q,q]=s*t2+c*t4
def rot_new(AT,p,q,c,s):
    app=AT[p,p]; aqq=AT[q,q]; apq=AT[q,p]
    for lo,hi in ((0,p),(p+1,q),(q+1,d)):
        x=AT[p,lo:hi].copy(); y=AT[q,lo:hi].copy()
        AT[p,lo:hi]=c*x-s*y; AT[q,lo:hi]=s*x+c*y
    t1=c*app-s*apq; t2=s*app+c*apq; t3=c*apq-s*aqq; t4=s*apq+c*aqq
    AT[p,p]=c*t1-s*t3; AT[p,q]=s*t1+c*t3
    AT[q,p]=c*t2-s*t4; AT[q,q]=s*t2+c*t4
for (p,q) in [(0,1),(0,2),(1,2)]:
    A=a0.copy()
    app=A[p,p]; aqq=A[q,q]; apq=A[p,q]
    theta=(aqq-app)/(2.0*apq); sign=1.0 if theta>=0 else -1.0
    t=sign/(abs(theta)+np.sqrt(theta*theta+1)); c=1/np.sqrt(t*t+1); s=t*c
    rot_old(A,p,q,c,s)
    AT=a0.T.copy(); rot_new(AT,p,q,c,s)
    print(p,q,"old\n",A,"\n new.T\n",AT.T)
