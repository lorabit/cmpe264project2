import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = "multiview/p3.JPG"
points2 = np.array([(992, 1784), (1248, 1684), (1464, 1804), (1612, 1340), (1620, 1596), (2696, 1644), (2724, 1384), (368, 1336), (592, 1320), (552, 992), (1128, 764), (1432, 812), (1640, 736), (1948, 752), (2360, 508), (2460, 504), (2620, 784)])
points1 = np.array([(948, 1588), (1200, 1516), (1452, 1576), (1536, 1180), (1540, 1448), (2560, 1444), (2564, 1184), (144, 1272), (384, 1256), (388, 880), (724, 1056), (1032, 1120), (1252, 1036), (1548, 1036), (1936, 788), (2028, 776), (2184, 1040)])

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

rr = -U.transpose()[2]
# t = np.matmul(np.matmul(np.matmul(U,W),S),U.transpose())
R = np.dot(np.dot(U,W),VT)

print("R = ")
print(R)
print("t = ")
print(rr)

rl = VT[2]
# print("rl = ")
# print(rl)
# print("Rrlx - E = ")
# print(np.matmul(R,[[0,-rl[2],rl[1]],[rl[2],0,-rl[0]],[-rl[1],rl[0],0]])-emtx)
# print(np.matmul(R.transpose(),R))


img=cv2.imread(img1)
imgplot = plt.imshow(img)

def depth(xl,xr,K,M,R,rr,rl):
	pr = np.dot(M,bar(xr))
	pl = np.dot(M,bar(xl))
	return f*np.dot(f*R[0] - pr[0]*R[2], rl)/np.dot(f*R[0] - pr[0]*R[2], pl)

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
		u = np.matmul(R,u) + rr
		print(u)
		u = np.dot(K, u)
		print(xl)
		print(u[0]/u[2],u[1]/u[2])
		plt.plot([u[0]/u[2]],[u[1]/u[2]],"x",color = 'r')


plt.autoscale(enable=True, axis='both', tight=True)
# plt.show()
