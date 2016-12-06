import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = "multiview/p3.JPG"
# points2 = np.array([(948, 1584), (1204, 1512), (1456, 1576), (140, 1280), (384, 1256), (1540, 1444), (2556, 1444), (2560, 1188), (1544, 1180), (1224, 1668), (1552, 1040), (724, 1048), (1036, 300), (524, 384), (560, 680), (1936, 784), (2024, 784), (2188, 1040)])
# points1 = np.array([(704, 1496), (1000, 1440), (1152, 1520), (260, 1184), (448, 1180), (1380, 1392), (2456, 1448), (2468, 1164), (1364, 1140), (852, 1592), (2036, 928), (1224, 940), (1544, 188), (1108, 300), (1140, 584), (2432, 660), (2524, 656), (2712, 940)])

# p2 
# points2 = np.array([(996, 1792), (1256, 1684), (1468, 1808), (128, 1192), (580, 1316), (632, 688), (1128, 768), (1616, 1348), (2732, 1392), (2704, 1648), (1620, 1604), (1644, 744), (1944, 756), (160, 148), (372, 164), (952, 48), (1380, 392)])
# p1 
# points1 = np.array([(708, 1500), (1004, 1444), (1152, 1512), (128, 1156), (444, 1180), (784, 896), (1224, 940), (1368, 1144), (2464, 1168), (2456, 1440), (1376, 1396), (1736, 928), (2040, 932), (412, 436), (600, 452), (1104, 304), (1488, 588)])



# p1 
points1 = np.array([(704, 1496), (1000, 1440), (1152, 1520), (260, 1184), (448, 1180), (1380, 1392), (2456, 1448), (2468, 1164), (1364, 1140), (852, 1592), (2036, 928), (1224, 940), (1544, 188), (1108, 300), (1140, 584), (2432, 660), (2524, 656), (2712, 940)])
# p3 
points2 = np.array([(948, 1584), (1204, 1512), (1456, 1576), (140, 1280), (384, 1256), (1540, 1444), (2556, 1444), (2560, 1188), (1544, 1180), (1224, 1668), (1552, 1040), (724, 1048), (1036, 300), (524, 384), (560, 680), (1936, 784), (2024, 784), (2188, 1040)])


# p2 
# points1 = np.array([(992, 1784), (1248, 1684), (1464, 1804), (1612, 1340), (1620, 1596), (2696, 1644), (2724, 1384), (368, 1336), (592, 1320), (552, 992), (1128, 764), (1432, 812), (1640, 736), (1948, 752), (2360, 508), (2460, 504), (2620, 784)])
# p3 
# points2 = np.array([(948, 1588), (1200, 1516), (1452, 1576), (1536, 1180), (1540, 1448), (2560, 1444), (2564, 1184), (144, 1272), (384, 1256), (388, 880), (724, 1056), (1032, 1120), (1252, 1036), (1548, 1036), (1936, 788), (2028, 776), (2184, 1040)])



emtx,emask = cv2.findEssentialMat(points1,points2)
fmtx,fmask = cv2.findFundamentalMat(points1,points2)
print("E =")
print(emtx)

U,S,VT = np.linalg.svd(emtx)
print("U = ")
print(U)

print("VT = ")
print(VT)

W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

# emtx = np.dot(np.dot(U,[[1,0,0],[0,1,0],[0,0,0]]),VT)
K = [[2771.214599609375, 0.0, 1387.954468694952], [0.0, 2767.546875, 937.6326279896311], [0.0, 0.0, 1.0]]
# K = [[2.83973773e+03, 0.00000000e+00, 1.38947081e+03], [0.00000000e+00, 2.83695500e+03, 9.36984169e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
M = np.linalg.inv(K)

f = K[0][0]

# rr = -U.transpose()[2]
rr = np.matmul(np.matmul(np.matmul(U,W),S),U.transpose())
R = np.dot(np.dot(U,W.transpose()),VT)

print("R = ")
print(R)
print("t = ")
print(rr)

VT_transpose = VT.transpose()
rl = VT_transpose[2]
# print("rl = ")
print("rl = ")
print(rl)
# print("Rrlx - E = ")
# print(np.matmul(R,[[0,-rl[2],rl[1]],[rl[2],0,-rl[0]],[-rl[1],rl[0],0]])-emtx)
# print(np.matmul(R.transpose(),R))


img=cv2.imread(img1)
imgplot = plt.imshow(img)

def depth(xl,xr,K,M,R,rr,rl):
	pr = np.dot(M,bar(xr))
	pl = np.dot(M,bar(xl))
	return f*np.dot(f*R[0] - pr[0]*R[2], rr)/np.dot(f*R[0] - pr[0]*R[2], pl)

def bar(v):
	u = v+ [1]
	return u

def p2c(pixel, M, f):
	v = np.dot(M,bar(pixel))
	v = v*f/v[2]
	return v

for i in range(len(emask)):
	if emask[i] == 1:
		xl = points1[i].tolist()
		xr = points2[i].tolist()
		print(bar(xl))
		z = np.dot(np.dot(bar(xr),fmtx),bar(xl))
		print("zf = "+str(z))
		# z = np.dot(np.dot(p2c(xl, M, f),emtx),p2c(xr, M, f))
		# print("ze = "+str(z))
		plt.plot(xl[0],xl[1],"o",color = 'y')
		print("depth = "+str(depth(xl, xr, K, M, R, rr, rl)))


		# u = np.linalg.inv(R).transpose()

		u = p2c(xr, M, f)
		print(u)
		u = np.dot(np.linalg.inv(R),u-rr)
		print(u)
		u = np.dot(K, u)
		print(xl)
		print(u[0]/u[2],u[1]/u[2])
		plt.plot([u[0]/u[2]],[u[1]/u[2]],"x",color = 'r')


plt.autoscale(enable=True, axis='both', tight=True)
plt.show()
