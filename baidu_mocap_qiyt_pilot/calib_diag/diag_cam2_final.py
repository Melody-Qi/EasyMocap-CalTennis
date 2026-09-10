# -*- coding: utf-8 -*-
"""决定性检查：final/ 里 cam1/cam2/cam3 各自的位姿，是否还拟合自己的球场点？
单独对每路做 solvePnP，比较 'PnP 解' 与 'final 里的位姿'。"""
import sys, json, numpy as np, cv2
sys.path.insert(0,'/public/home/CS286/qiyt2023-CS286/pilot_3cam')
from triang_3cam_pilot import parse_opencv_yml
MY='/public/home/CS286/qiyt2023-CS286/pilot_3cam/calib_result/final'
kA=parse_opencv_yml(MY+'/intri.yml'); eA=parse_opencv_yml(MY+'/extri.yml')
C=jc=json.load(open('/public/home/CS286/qiyt2023-CS286/pilot_3cam/court_annotations.json'))
print('%-6s %6s %14s %14s %14s %12s' % ('cam','n','PnP重投影中位','final重投影中位','光心差(m)','旋转差(deg)'))
for i in [1,2,3]:
    ent=jc['cam%d'%i]
    pts=np.array([v['xyz'] for v in ent.values()],float)
    uvs=np.array([v['uv'] for v in ent.values()],float)
    K=np.array(kA['K_cam%d'%i],float)
    ok,rv,tv=cv2.solvePnP(pts,uvs,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
    Rp=cv2.Rodrigues(rv)[0]; Tp=tv.reshape(3,1)
    def repo(R,T,und=True):
        pr,_=cv2.projectPoints(pts, cv2.Rodrigues(R)[0], T, K, None)
        return np.median(np.linalg.norm(pr.reshape(-1,2)-uvs,axis=1))
    Rf=np.array(eA['Rot_cam%d'%i],float); Tf=np.array(eA['T_cam%d'%i],float).reshape(3,1)
    cp=(-Rp.T@Tp).ravel(); cf=(-Rf.T@Tf).ravel()
    ang=np.degrees(np.arccos(np.clip((np.trace(Rp.T@Rf)-1)/2,-1,1)))
    print('%-6s %6d %14.1f %14.1f %14.3f %12.2f' % (
        'cam%d'%i, len(pts), repo(Rp,Tp), repo(Rf,Tf), np.linalg.norm(cp-cf), ang))
print()
print('参考：光心世界坐标')
for i in [1,2,3]:
    ent=jc['cam%d'%i]
    pts=np.array([v['xyz'] for v in ent.values()],float)
    uvs=np.array([v['uv'] for v in ent.values()],float)
    K=np.array(kA['K_cam%d'%i],float)
    ok,rv,tv=cv2.solvePnP(pts,uvs,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
    cp=(-cv2.Rodrigues(rv)[0].T@tv.reshape(3,1)).ravel()
    Rf=np.array(eA['Rot_cam%d'%i],float); Tf=np.array(eA['T_cam%d'%i],float).reshape(3,1)
    cf=(-Rf.T@Tf).ravel()
    print('  cam%d  PnP=(%7.2f,%7.2f,%6.2f)   final=(%7.2f,%7.2f,%6.2f)' % (i,*cp,*cf))
