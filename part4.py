import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img1 = "multiview/p3.JPG"
points2 = np.array([(948, 1584), (1204, 1512), (1456, 1576), (140, 1280), (384, 1256), (1540, 1444), (2556, 1444), (2560, 1188), (1544, 1180), (1224, 1668), (1552, 1040), (724, 1048), (1036, 300), (524, 384), (560, 680), (1936, 784), (2024, 784), (2188, 1040)])
points1 = np.array([(704, 1496), (1000, 1440), (1152, 1520), (260, 1184), (448, 1180), (1380, 1392), (2456, 1448), (2468, 1164), (1364, 1140), (852, 1592), (2036, 928), (1224, 940), (1544, 188), (1108, 300), (1140, 584), (2432, 660), (2524, 656), (2712, 940)])

emtx,emask = cv2.findEssentialMat(points1,points2)
fmtx,fmask = cv2.findFundamentalMat(points2,points1)
# print("E =")
# print(emtx)

U,S,V = np.linalg.svd(emtx)
VT = V.T

if np.linalg.det(U)<0 and np.linalg.det(V)<0:
	print 'Error'
	exit(0)
else:
	if np.linalg.det(U)*np.linalg.det(V)<0:
		print 'Error <0'
		exit(0)

# print("U = ")
# print(U)

# print("VT = ")
# print(VT)

W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

# emtx = np.matmul(np.matmul(U,[[1,0,0],[0,1,0],[0,0,0]]),VT)
K = [[2771.214599609375, 0.0, 1387.954468694952], [0.0, 2767.546875, 937.6326279896311], [0.0, 0.0, 1.0]]
# K = [[2.83973773e+03, 0.00000000e+00, 1.38947081e+03], [0.00000000e+00, 2.83695500e+03, 9.36984169e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
M = np.linalg.inv(K)

f = K[0][0]


# rr = np.matmul(np.matmul(np.matmul(U,W),S),U.transpose())
R = np.matmul(np.matmul(U,W),VT.T).T
rr = U[2]
rl = VT[2]


print "R = ",R
print "RT*R = ", np.matmul(R.T,R)
print 'det(R) = ', np.linalg.det(R)
print "rr = ",rr
print "rl = ",rl


# print("rl = ")
print("rl = ")
print(rl)
# print("Rrlx - E = ")
# print(np.matmul(R,[[0,-rl[2],rl[1]],[rl[2],0,-rl[0]],[-rl[1],rl[0],0]])-emtx)
# print(np.matmul(R.transpose(),R))


img=cv2.imread(img1)
imgplot = plt.imshow(img)

def depth(xl,xr,K,M,R,rr,rl):
	pr = np.matmul(M,bar(xr))
	pl = np.matmul(M,bar(xl))
	return f*np.dot(f*R[0] - pr[0]*R[2], rr)/np.dot(f*R[0] - pr[0]*R[2], pl)

def bar(v):
	u = v+ [1]
	return u

def p2c(pixel, M, f):
	v = np.matmul(M,bar(pixel))
	v = v*f/v[2]
	return v


for i in range(len(emask)):
	if emask[i] == 1:
		xl = points1[i].tolist()
		xr = points2[i].tolist()
		# print(bar(xl))
		# z = np.matmul(np.matmul(bar(xr),fmtx),bar(xl))
		# print("zf = "+str(z))
		# z = np.matmul(np.matmul(p2c(xl, M, f),emtx),p2c(xr, M, f))
		# print("ze = "+str(z))
		plt.plot(xl[0],xl[1],"o",color = 'y')
		print("depth = "+str(depth(xl, xr, K, M, R, rr, rl)))


		# u = np.linalg.inv(R).transpose()

		u = p2c(xr, M, f)
		print(u)
		u = np.matmul(R,u)
		# u = np.matmul(np.linalg.inv(R),u)+300*rl
		print(u)
		u = np.matmul(K, u)
		print(xl)
		print(u[0]/u[2],u[1]/u[2])
		plt.plot([u[0]/u[2]],[u[1]/u[2]],"x",color = 'r')


plt.autoscale(enable=True, axis='both', tight=True)
plt.show()
