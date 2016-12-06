import cv2
import numpy as np
import os

 
mtx = np.array([[2.83973773e+03, 0.00000000e+00, 1.38947081e+03], [0.00000000e+00, 2.83695500e+03, 9.36984169e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.12244297, 0.20499093, 0.00035033, -0.00058871, -0.13275741]])
h,  w = 1856, 2784
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


print(mtx)
print(newcameramtx.tolist())

files = os.listdir('multiview')
for filename in files:
	if filename[-3:].lower()=='jpg':
		img = cv2.imread('multiview/'+filename)
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		x,y,w,h = roi
		print(roi)
		dst = dst[y:y+h, x:x+w]
		cv2.imwrite('undistorted/'+filename,dst)