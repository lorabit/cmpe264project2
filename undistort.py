import cv2
import numpy as np
import os


mtx = np.array([[  2.82828657e+03,0.00000000e+00,1.46260639e+03],[  0.00000000e+00,2.86745094e+03,9.78000755e+02],[  0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist = np.array([[ -2.59349408e-01,2.40450821e+00,2.47497177e-03,4.83354522e-03,-1.18401952e+01]])
h,  w = 1856, 2784
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print(mtx)
print(newcameramtx)

files = os.listdir('multiview')
for filename in files:
	if filename[-3:].lower()=='jpg':
		img = cv2.imread('multiview/'+filename)
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		x,y,w,h = roi
		# dst = dst[y:y+h, x:x+w]
		cv2.imwrite('undistorted/'+filename,dst)