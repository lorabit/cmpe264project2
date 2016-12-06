import numpy as np
from scipy.optimize import minimize
import math
import cv2

r1_2_to_1 = np.array(1,1,1)
r2_3_to_2 = np.array(2,2,2)
r1_3_to_1 = np.array(3,3,3)

R1_2 = np.array()
R2_3 = np.array()
R1_3 = np.array()

r2_3_to_1 = np.dot(R1_2.transpose(), r2_3_to_2)
r3_1_to_1 = -r1_3_to_1

r1_2_to_1_unit = r1_2_to_1 / math.sqrt( np.pow(r1_2_to_1[0],2) + np.pow(r1_2_to_1[1],2) + np.pow(r1_2_to_1[2],2) )
r2_3_to_1_unit = r2_3_to_1 / math.sqrt( np.pow(r2_3_to_1[0],2) + np.pow(r2_3_to_1[1],2) + np.pow(r2_3_to_1[2],2) )
r3_1_to_1_unit = r3_1_to_1 / math.sqrt( np.pow(r3_1_to_1[0],2) + np.pow(r3_1_to_1[1],2) + np.pow(r3_1_to_1[2],2) )



def rosen(beta, gama):
	
	return np.pow(r1_2_to_1_unit[0] + beta*r2_3_to_1_unit[0] + gama*r3_1_to_1_unit[0], 2) + np.pow(r1_2_to_1_unit[1] + beta*r2_3_to_1_unit[1] + gama*r3_1_to_1_unit[1], 2) + np.pow(r1_2_to_1_unit[2] + beta*r2_3_to_1_unit[2] + gama*r3_1_to_1_unit[2], 2)


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen, x0, methord = 'nelder-mead', options = {'xtol': le-8, 'disp': True})

beta = res.x[0]
gama = res.x[1]

print 'beta = ' + beta
print 'gama = ' + gama

mtx_with_beta_gama = r1_2_to_1_unit + beta * r2_3_to_1_unit + gama * r3_1_to_1_unit
mtx_wout_beta_gama = r1_2_to_1_unit + r2_3_to_1_unit + r3_1_to_1_unit

vec_len_mtx_with_beta_gama = math.sqrt( np.pow(mtx_with_beta_gama[0], 2) + np.pow(mtx_with_beta_gama[1], 2) + np.pow(mtx_with_beta_gama[2], 2) )
vec_len_mtx_wout_beta_gama = math.sqrt( np.pow(mtx_wout_beta_gama[0], 2) + np.pow(mtx_wout_beta_gama[1], 2) + np.pow(mtx_wout_beta_gama[2], 2) )

print 'r12 + beta * r23 + gama * r31 = ' + vec_len_mtx_with_beta_gama
print 'r12 + r23 + r31 = ' + vec_len_mtx_wout_beta_gama
