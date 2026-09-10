# -*- coding: utf-8 -*-
"""对每路球场点做 RANSAC PnP，看是否只有少数错标点。"""
import sys, json, numpy as np, cv2
sys.path.insert(0,'/public/home/CS286/qiyt2023-CS286/pilot_3cam')
from triang_3cam_pilot import parse_opencv_yml
MY='/public/home/CS286/qiyt2023-CS286/pilot_3cam/calib_result/final'
kA=parse_opencv_yml(MY+'/intri.yml')
jc=json.load(open('/public/home/CS286/qiyt2023-CS286/pilot_3cam/court_annotations.json'))
for i in [1,2,3]:
    ent=jc['cam%d'%i]
    names=list(ent.keys())
    pts=np.array([ent[n]['xyz'] for n in names],float)
    uvs=np.array([ent[n]['uv'] for n in names],float)
    K=np.array(kA['K_cam%d'%i],float)
    ok,rv,tv,inl=cv2.solvePnPRansac(pts,uvs,K,None,reprojectionError=8.0,iterationsCount=5000,confidence=0.999)
    n_in = 0 if inl is None else len(inl)
    print('=== cam%d: %d 点, RANSAC inlier %d ===' % (i,len(names),n_in))
    if n_in:
        R=cv2.Rodrigues(rv)[0]; T=tv.reshape(3,1)
        pr,_=cv2.projectPoints(pts,rv,T,K,None); pr=pr.reshape(-1,2)
        err=np.linalg.norm(pr-uvs,axis=1)
        idx=inl.ravel()
        print('  inlier 残差: med=%.2f max=%.2f px' % (np.median(err[idx]), err[idx].max()))
        bad=[(names[j], err[j]) for j in range(len(names)) if j not in set(idx.tolist())]
        print('  outlier 点 (%d):' % len(bad))
        for n,e in bad: print('     %-14s err=%9.1f px   uv=%s' % (n, e, ent[n]['uv']))
        print('  RANSAC 光心: (%7.2f,%7.2f,%6.2f)' % tuple((-R.T@T).ravel()))
