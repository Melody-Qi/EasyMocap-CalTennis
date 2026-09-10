# -*- coding: utf-8 -*-
import sys, numpy as np, cv2
sys.path.insert(0,'/public/home/CS286/qiyt2023-CS286/pilot_3cam')
from triang_3cam_pilot import parse_opencv_yml
MY='/public/home/CS286/qiyt2023-CS286/pilot_3cam/calib_result/final'
WU='/public/home/CS286/qiyt2023-CS286/EasyMocap/baidu_mocap_20260831/work'
kA=parse_opencv_yml(MY+'/intri.yml'); eA=parse_opencv_yml(MY+'/extri.yml')
fsB_i=cv2.FileStorage(WU+'/intri.yml', cv2.FILE_STORAGE_READ)
fsB_e=cv2.FileStorage(WU+'/extri_revised.yml', cv2.FILE_STORAGE_READ)
def gb(fs,n):
    nd=fs.getNode(n); return nd.mat() if not nd.empty() else None
print('%-6s %12s %14s %12s %18s' % ('cam','f_A(fix)','f_B(学长)','dist_B','光心差 m'))
res={}
for i in [1,2,3]:
    KA=np.array(kA['K_cam%d'%i],float); KB=gb(fsB_i,'K_tennis_camera_%d'%i)
    dB=gb(fsB_i,'dist_tennis_camera_%d'%i)
    RA=np.array(eA['Rot_cam%d'%i],float); TA=np.array(eA['T_cam%d'%i],float).reshape(3,1)
    RB=gb(fsB_e,'Rot_tennis_camera_%d'%i); TB=gb(fsB_e,'T_tennis_camera_%d'%i)
    cA=(-RA.T@TA).ravel(); cB=(-RB.T@TB).ravel()
    res[i]=(cA,cB,RA,RB)
    print('%-6s %12.1f %14.1f %12s %18.3f' % ('cam%d'%i, KA[0,0], KB[0,0],
        'yes k1=%.3f'%dB[0,0] if dB is not None else 'none', np.linalg.norm(cA-cB)))
print()
print('光心世界坐标 (m):')
print('%-6s %-26s %-26s' % ('cam','A = final/ (当前在用)','B = 学长 revised (9/9 23:42)'))
for i in [1,2,3]:
    cA,cB,_,_=res[i]
    print('%-6s (%7.2f,%8.2f,%6.2f)  (%7.2f,%8.2f,%6.2f)' % ('cam%d'%i,*cA,*cB))
print()
print('旋转差异 (deg):')
for i in [1,2,3]:
    _,_,RA,RB=res[i]
    print('  cam%d  %6.2f°' % (i, np.degrees(np.arccos(np.clip((np.trace(RA.T@RB)-1)/2,-1,1)))))
