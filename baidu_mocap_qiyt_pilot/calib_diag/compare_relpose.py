# -*- coding: utf-8 -*-
"""用【不变量】判断两套标定是真不同还是只差世界系：比较相机对之间的相对位姿。"""
import sys, numpy as np, cv2
sys.path.insert(0,'/public/home/CS286/qiyt2023-CS286/pilot_3cam')
from triang_3cam_pilot import parse_opencv_yml
MY='/public/home/CS286/qiyt2023-CS286/pilot_3cam/calib_result/final'
WU='/public/home/CS286/qiyt2023-CS286/EasyMocap/baidu_mocap_20260831/work'
kA=parse_opencv_yml(MY+'/intri.yml'); eA=parse_opencv_yml(MY+'/extri.yml')
fi=cv2.FileStorage(WU+'/intri.yml', cv2.FILE_STORAGE_READ)
fe=cv2.FileStorage(WU+'/extri_revised.yml', cv2.FileStorage_READ)
def gb(fs,n):
    nd=fs.getNode(n); return nd.mat() if not nd.empty() else None
RA={};TA={};RB={};TB={}
for i in [1,2,3]:
    RA[i]=np.array(eA['Rot_cam%d'%i],float); TA[i]=np.array(eA['T_cam%d'%i],float).reshape(3,1)
    RB[i]=gb(fe,'Rot_tennis_camera_%d'%i); TB[i]=gb(fe,'T_tennis_camera_%d'%i).reshape(3,1)
def relpose(R,T,i,j):
    # j 相机在 i 相机坐标系中的位姿
    Ri,Rj=R[i],R[j]; Ti,Tj=T[i],T[j]
    Rrel=Rj@Ri.T; Trel=Tj-Rrel@Ti
    return Rrel,Trel
print('%-12s %14s %14s %12s' % ('pair (i->j)','|t| A (m)','|t| B (m)','角差 deg'))
for i,j in [(1,2),(1,3),(2,3)]:
    Ra,Ta=relpose(RA,TA,i,j); Rb,Tb=relpose(RB,TB,i,j)
    ang=np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1)))
    print('cam%d->cam%d %14.3f %14.3f %12.2f' % (i,j,np.linalg.norm(Ta),np.linalg.norm(Tb),ang))
print()
print('结论判据：若 |t| 与角差都接近 0 -> 只差世界系定义；否则是真不同。')
