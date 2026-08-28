import sys, json, numpy as np, cv2
root, c1, c2, nf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def load_cam(cam):
    fs=cv2.FileStorage(f'{root}/intri.yml', cv2.FILE_STORAGE_READ); K=fs.getNode(f'K_{cam}').mat(); fs.release()
    fs=cv2.FileStorage(f'{root}/extri.yml', cv2.FILE_STORAGE_READ); R=fs.getNode(f'Rot_{cam}').mat(); T=fs.getNode(f'T_{cam}').mat().reshape(3); fs.release()
    return K,R,T
def Pmat(K,R,T): return K@np.hstack([R,T.reshape(3,1)])
def triang(uv1,uv2,P1,P2):
    A=np.vstack([uv1[0]*P1[2]-P1[0],uv1[1]*P1[2]-P1[1],uv2[0]*P2[2]-P2[0],uv2[1]*P2[2]-P2[1]])
    _,_,Vt=np.linalg.svd(A); X=Vt[-1]; return X[:3]/X[3]
def repro(X,P):
    xh=P@np.append(X,1.0); return xh[:2]/xh[2]
Ps=[Pmat(*load_cam(c)) for c in (c1,c2)]
ann=[json.load(open(f'{root}/annots/{c}/{nf:06d}.json'))['annots'] for c in (c1,c2)]
print(f'{c1} persons={len(ann[0])} {c2} persons={len(ann[1])}')
best=None
for i,a in enumerate(ann[0]):
    for j,b in enumerate(ann[1]):
        ka=np.array(a['keypoints']); kb=np.array(b['keypoints']); errs=[]; Xs=[]
        for k in range(25):
            if ka[k,2]>0.3 and kb[k,2]>0.3:
                X=triang(ka[k,:2],kb[k,:2],Ps[0],Ps[1]); Xs.append(X)
                errs.append(np.linalg.norm(repro(X,Ps[0])-ka[k,:2])+np.linalg.norm(repro(X,Ps[1])-kb[k,:2]))
        if len(Xs)>=10:
            e=np.mean(errs)
            if best is None or e<best[0]: best=(e,np.array(Xs),i,j)
if best is None:
    print('no valid pair'); sys.exit()
e,Xs,i,j=best
print(f'best {c1}#{i} x {c2}#{j}: repro={e:.2f}px height={Xs[:,2].max()-Xs[:,2].min():.2f}m  Xmean={Xs[:,0].mean():.2f}')
