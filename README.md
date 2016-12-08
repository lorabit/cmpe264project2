# Requirements
1. OpenCV with contrib
2. numpy
3. sklearn
4. matplotlib
5. Python 2.7
6. scipy

# Part 1

In part 1, we import numpy to do the numeric work on this part.


# Part 3


Fill the intrsinc matrix, undistort parameters, size of image in the begining of undistort.py. Then run python undistort.py to undistort images in multiview folder and save into undistorted folder. 

Use ```python click.py --image [PATH to Image]``` to start picking pixels. Press any key to stop picking and you will see the array of picked pixels in the console. 

Fill the arrays obtained from previous step into part3.py, then run python part3.py.

# Part 4

Fill the pixel arrays obtained from previous part in the begining of part4.py and run ```python part4.py``` to see the reprojection result. 

# Part 5

# Part 5
In part 5, we import the minimize function from the scipy.optimize to do the minimization on the vector we calculate, and the function give us the result of the ideal value of beta and gamma as shown in part5.py 