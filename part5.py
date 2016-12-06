import numpy as np
from scipy.optimize import minimize
import math
import cv2

r1_2_to_1 = np.array([6.46921084e-01,   7.62556951e-01,   8.21027759e-05])
r2_3_to_2 = np.array([3.25635439e-01,  -9.45495385e-01,  -1.95901070e-04])
r1_3_to_1 = np.array([ -9.98006457e-01,  6.31118826e-02,  -3.40855044e-05])

R1_2 = np.array([[ -1.44081746e-01,   9.89565783e-01,   1.10346486e-04], [  9.89565786e-01,   1.44081735e-01,   9.55294090e-05], [  7.86337212e-05,   1.22959151e-04,  -9.99999989e-01]])
R2_3 = np.array([[9.27941616e-01,   -3.72725544e-01,  -1.62471718e-04], [  3.72725559e-01,   9.27941621e-01,   7.24020613e-05], [  1.23778171e-04,  -1.27742248e-04,   9.99999984e-01]])
R1_3 = np.array([[ -9.95709372e-01,   9.25356182e-02,   7.71657097e-05], [ -9.25356222e-02,  -9.95709373e-01,  -4.98016054e-05], [ -7.22261981e-05,   5.67285022e-05,  -9.99999996e-01]])

r2_3_to_1 = np.dot(R1_2.transpose(), r2_3_to_2)
r3_1_to_1 = -r1_3_to_1


# unit form the r 
r1_2_to_1_unit = r1_2_to_1 / math.sqrt( pow(r1_2_to_1[0],2) + pow(r1_2_to_1[1],2) + pow(r1_2_to_1[2],2) )
r2_3_to_1_unit = r2_3_to_1 / math.sqrt( pow(r2_3_to_1[0],2) + pow(r2_3_to_1[1],2) + pow(r2_3_to_1[2],2) )
r3_1_to_1_unit = r3_1_to_1 / math.sqrt( pow(r3_1_to_1[0],2) + pow(r3_1_to_1[1],2) + pow(r3_1_to_1[2],2) )



def rosen(x):
	
	return pow(r1_2_to_1_unit[0] + x[0]*r2_3_to_1_unit[0] + x[1]*r3_1_to_1_unit[0], 2) + pow(r1_2_to_1_unit[1] + x[0]*r2_3_to_1_unit[1] + x[1]*r3_1_to_1_unit[1], 2) + pow(r1_2_to_1_unit[2] + x[0]*r2_3_to_1_unit[2] + x[1]*r3_1_to_1_unit[2], 2)


x0 = np.array([1.3, 0.7])

res = minimize(rosen, x0, method = 'nelder-mead', options={'xtol': 1e-8, 'disp': True})

beta = res.x[0]
gama = res.x[1]

print 'beta = ' 
print(beta)
print 'gama = '
print(gama)

mtx_with_beta_gama = r1_2_to_1_unit + beta * r2_3_to_1_unit + gama * r3_1_to_1_unit
mtx_wout_beta_gama = r1_2_to_1_unit + r2_3_to_1_unit + r3_1_to_1_unit

vec_len_mtx_with_beta_gama = math.sqrt( pow(mtx_with_beta_gama[0], 2) + pow(mtx_with_beta_gama[1], 2) + pow(mtx_with_beta_gama[2], 2) )
vec_len_mtx_wout_beta_gama = math.sqrt( pow(mtx_wout_beta_gama[0], 2) + pow(mtx_wout_beta_gama[1], 2) + pow(mtx_wout_beta_gama[2], 2) )

print '|| r12 + beta * r23 + gama * r31 || = '
print(vec_len_mtx_with_beta_gama)
print '|| r12 + r23 + r31 || = '
print(vec_len_mtx_wout_beta_gama)


