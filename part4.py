import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = "multiview/p3.JPG"
points1 = np.array([(952, 1612), (1200, 1540), (1460, 1612), (1444, 1068), (1548, 1160), (2568, 1200), (1548, 1472), (720, 1068), (1028, 1136), (1048, 756), (1648, 736), (1560, 1056), (524, 388), (556, 1240), (544, 696), (136, 1296), (388, 1268), (384, 892)])
points2 = np.array([(1000, 1812), (1248, 1712), (1464, 1832), (1648, 1052), (1628, 1328), (2728, 1392), (1632, 1624), (1124, 768), (1444, 828), (1456, 440), (2052, 444), (1960, 768), (952, 44), (1012, 884), (988, 348), (368, 1352), (588, 1336), (552, 1004)])

emtx,emask = cv2.findEssentialMat(points2,points1)
print("E =")
print(emtx)

U,S,VT = np.linalg.svd(emtx)
print("U = ")
print(U)

print("VT = ")
print(VT)

W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

emtx = np.dot(np.dot(U,[[1,0,0],[0,1,0],[0,0,0]]),VT)

t = U.transpose()[2]
# t = np.matmul(np.matmul(np.matmul(U,W),S),U.transpose())
R = np.dot(np.dot(U,W),VT)

print("R = ")
print(R)
print("t = ")
print(t)

rl = VT[2]
print("rl = ")
print(rl)
print("Rrlx - E = ")
print(np.matmul(R,[[0,-rl[2],rl[1]],[rl[2],0,-rl[0]],[-rl[1],rl[0],0]])-emtx)
# print(np.matmul(R.transpose(),R))


img=cv2.imread(img1)
imgplot = plt.imshow(img)

for i in range(len(emask)):
	if emask[i] == 1:
		plt.plot([points1[i][0]],points1[i][1],"o",color = 'y')
		v = np.array([points2[i][0],points2[i][1],1])
		u = np.matmul(R,v) + t
		plt.plot([u[0]/u[2]],[u[1]/u[2]],"x",color = 'r')
		print([u[0]/u[2],u[1]/u[2]])


plt.autoscale(enable=True, axis='both', tight=True)
plt.show()
