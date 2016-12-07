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
# M = np.linalg.inv(K)
f = K[0][0]	
M = [[1,0,-K[0][2]],[0,1,-K[1][2]],[0,0,f]]
# print K
# print M
# print np.matmul(K,M)


rl = -VT[2]


def depth(xl,xr):
	# print R
	pr = p2c(xr)
	pl = p2c(xl)
	# print pr, pl
	d = f*np.dot(f*R[0] - pr[0]*R[2], rl)/np.dot(f*R[0] - pr[0]*R[2], pl)
	# print trans(pr)
	# print c2p(trans(pr)), xl
	return d

def trans(p):
	return np.dot(R,p)

def bar(v):
	u = v+ [1]
	# print v,u
	return u

def c2p(p):
	v = np.dot(K,p)
	return [v[0]/v[2],v[1]/v[2]]

def p2c(pixel):
	v = np.dot(M,bar(pixel))
	v = v*f/v[2]
	return v


def plot(R,rr):
	img=cv2.imread(img1)
	imgplot = plt.imshow(img)



	for i in range(len(emask)):
		if emask[i] == 1:
			xl = points1[i].tolist()
			xr = points2[i].tolist()

			plt.plot(xl[0],xl[1],"o",color = 'y')
			print("depth = "+str(depth(xl, xr)))

			u = c2p(trans(p2c(xr)))
			print xl,u
			plt.plot(u[0],u[1],"x",color = 'r')


	plt.autoscale(enable=True, axis='both', tight=True)
	plt.show()


R1, R2 = np.matmul(np.matmul(U,W),V), np.matmul(np.matmul(U,W.T),V)


for R,rr in (R1, U.T[2]),(R2,U.T[2]),(R1, -U.T[2]),(R2,-U.T[2]):
	print 'det(R) = ',np.linalg.det(R)
	# rl = np.dot(R,rr)
	valid = True
	for i in range(len(emask)):
		if emask[i] == 1:
			xl = points1[i].tolist()
			xr = points2[i].tolist()
			d = depth(xl, xr)
			print d
			if d<0:
				valid = False
				# break
	plot(R,rr)
	exit(0)
	if valid:
		print R,rr
			




# print "R = ",R
# print "RT*R = ", np.matmul(R.T,R)
# print 'det(R) = ', np.linalg.det(R)
# print "rr = ",rr
# print "rl = ",rl


# # print("rl = ")
# print("rl = ")
# print(rl)
# print("Rrlx - E = ")
# print(np.matmul(R,[[0,-rl[2],rl[1]],[rl[2],0,-rl[0]],[-rl[1],rl[0],0]])-emtx)
# print(np.matmul(R.transpose(),R))



