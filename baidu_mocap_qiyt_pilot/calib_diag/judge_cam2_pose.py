# -*- coding: utf-8 -*-
"""决定性裁判：cam1+cam3 位姿可信（各自 RANSAC 残差 4.09/2.57px），
用它们三角化出人体 3D（与球场点无关），投影到 cam2 的 ViTPose 2D。
谁的 cam2 位姿误差小，谁对。"""
import sys, os, numpy as np, torch, cv2
sys.path.insert(0,'/public/home/CS286/qiyt2023-CS286/pilot_3cam')
from triang_3cam_pilot import parse_opencv_yml, triang_nview
MY='/public/home/CS286/qiyt2023-CS286/pilot_3cam/calib_result/final'
BASE='/public/home/CS286/qiyt2023-CS286/GVHMR/outputs'
kA=parse_opencv_yml(MY+'/intri.yml'); eA=parse_opencv_yml(MY+'/extri.yml')
K={i:np.array(kA['K_cam%d'%i],float) for i in [1,2,3]}
def P(i,R=None,T=None):
    if R is None: R=np.array(eA['Rot_cam%d'%i],float); T=np.array(eA['T_cam%d'%i],float).reshape(3,1)
    return K[i]@np.hstack([R,T])
P1=P(1); P3=P(3)
vps={c: torch.load(os.path.join(BASE,f'baidu_pilot_15_30s_tennis_camera_{c}',f'tennis_camera_{c}','preprocess','vitpose.pt')).numpy() for c in [1,2,3]}
N=vps[1].shape[0]
# 候选 cam2 位姿
cands={}
cands['A_final(fix_cam2人体解)']=(np.array(eA['Rot_cam2'],float), np.array(eA['T_cam2'],float).reshape(3,1))
# B: RANSAC 球场解（重算）
import json
jc=json.load(open('/public/home/CS286/qiyt2023-CS286/pilot_3cam/court_annotations.json'))
ent=jc['cam2']; names=list(ent.keys())
pts=np.array([ent[n]['xyz'] for n in names],float); uvs=np.array([ent[n]['uv'] for n in names],float)
ok,rv,tv,inl=cv2.solvePnPRansac(pts,uvs,K[2],None,reprojectionError=8.0,iterationsCount=5000,confidence=0.999)
RB=cv2.Rodrigues(rv)[0]; TB=tv.reshape(3,1)
cands['B_RANSAC球场解']=(RB,TB)
print('%-26s %10s %10s %10s %12s' % ('cam2 位姿候选','中位px','均值px','p90','有效点数'))
for name,(R2,T2) in cands.items():
    P2=K[2]@np.hstack([R2,T2])
    errs=[]; npts=0
    for f in range(N):
        v=[vps[1][f],vps[3][f],vps[2][f]]
        for j in range(17):
            if v[0][j,2]<0.5 or v[1][j,2]<0.5: continue
            X=triang_nview([P1,P3],[tuple(v[0][j,:2]),tuple(v[1][j,:2])],[1,1],2)
            if X is None: continue
            if v[2][j,2]<0.5: continue
            x=P2@np.append(X[:3],1.0)
            if abs(x[2])<1e-9: continue
            uv=x[:2]/x[2]
            e=float(np.hypot(uv[0]-v[2][j,0], uv[1]-v[2][j,1]))
            if e<3000: errs.append(e); npts+=1
    errs=np.array(errs)
    print('%-26s %10.1f %10.1f %10.1f %12d' % (name, np.median(errs), errs.mean(), np.percentile(errs,90), npts))
    print('      光心=(%7.2f,%7.2f,%6.2f)' % tuple((-R2.T@T2).ravel()))
