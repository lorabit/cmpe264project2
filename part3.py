import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

h,  w = 1856, 2784
h2 , w2 = 9.19567171e+02, 1.43250832e+03 
img1 = "multiview/p3.JPG"
img2 = "multiview/p2.JPG"
points1 = np.array([(952, 1612), (1200, 1540), (1460, 1612), (1444, 1068), (1548, 1160), (2568, 1200), (1548, 1472), (720, 1068), (1028, 1136), (1048, 756), (1648, 736), (1560, 1056), (524, 388), (556, 1240), (544, 696), (136, 1296), (388, 1268), (384, 892)])
points2 = np.array([(1000, 1812), (1248, 1712), (1464, 1832), (1648, 1052), (1628, 1328), (2728, 1392), (1632, 1624), (1124, 768), (1444, 828), (1456, 440), (2052, 444), (1960, 768), (952, 44), (1012, 884), (988, 348), (368, 1352), (588, 1336), (552, 1004)])

mtx,mask = cv2.findFundamentalMat(points1,points2)
print(mtx)
print(mask)

inliersX = []
inliersY = []

outliersX = []
outliersY = []

for i in range(len(mask)):
	if mask[i] == 1:
		inliersX += [points1[i][0]]
		inliersY += [points1[i][1]]
		# break
	else:
		outliersX += [points1[i][0]]
		outliersY += [points1[i][1]]

img=cv2.imread(img1)
imgplot = plt.imshow(img)
plt.plot(inliersX,inliersY,"o",color = 'y')
plt.plot(outliersX,outliersY,"o",color = 'gray')
for i in range(len(mask)):
	if mask[i] == 1:
		print(np.dot(np.dot(np.array([points1[i][0],points1[i][1],1])),mtx),np.array([points2[i][0],points2[i][1],1]))
		a,b,c  = np.matmul(mtx,np.array([points2[i][0],points2[i][1],1]))
		px = []
		py = []
		print(a,b,c)
		x = 0
		y = -1/b*(a*x+c)
		if y>=0 and y<=h:
			px += [x]
			py += [y]
		x = w
		y = -1/b*(a*x+c)
		if y>=0 and y<=h:
			px += [x]
			py += [y]
		y = 0
		x = -1/a*(b*x+c)
		if x>=0 and x<=w:
			px += [x]
			py += [y]
		y = h
		x = -1/a*(b*x+c)
		if x>=0 and x<=w:
			px += [x]
			py += [y]
		if len(px)==2:
			plt.plot(px,py)
		# break
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()