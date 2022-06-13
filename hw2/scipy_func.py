import scipy
import numpy as np
from scipy import linalg # scipy require to import by ourselves
from scipy.optimize import nnls
from scipy.io import savemat
from scipy.ndimage import rotate

def w1(A,b):
    """
    Solves the linear equation set a * x = b for the unknown x for square a matrix.
    Input:
    - A: A numpy array with shape [M,N]
    - b: A numpy array with shape [M,]

    Returns:
    - Solved result X with shape [N,]

    Hint: Use linalg.solve
    """
    return linalg.solve(A,b)

def w2(A):
    """
    Compute the inverse of a matrix   
     Input:
    - A: A numpy array with shape [N,N]

    Returns:
    - Inverse of A with shape [N,N]

    Hint: Use linalg.inv
    """
    return linalg.inv(A)

def w3(A,b):
    """
    Solve argmin_x || Ax - b ||_2 for x>=0. 

    Input:
    - A: A numpy array with shape [M,N]
    - b: A numpy array with shape [M,]

    Returns:
    - Solution vector x with shape [N,]

    Hint: Use scipy.optimize.nnls
    """
    return nnls(A,b)[0] # nnls return a tuple, we need the first element, which is a ndarray

def w4(a):
    """
    Save a dictionary of names and arrays into a .mat file.

    Input:
    - a: Dictionary from which to save matfile variables.
    e.g: a = {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), 'label': 'experiment'}

    Hint: Use scipy.io.savemat
    """
    midc = {"a":a, "label":"experiment"}
    savemat("w4_out.mat",midc)
    return None

def w5(A):
    """
    Rotate an array for 180 degree on the 0-th axis.

    The array is rotated in the plane defined by the two axes given by the axes parameter using spline interpolation of the requested order.
    
    Input:
    - A: The input array as an image

    Returns:
    - Rotated image

    Hint: Use scipy.ndimage.rotate
    """
    return rotate(A, angle = 180) # assume reshape = True

# The code below are testcases
from scipy import ndimage, misc
import matplotlib.pyplot as plt
if __name__ == "__main__":
    array_t11 = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    array_t12 = np.array([2,4,-1])
    print("w1",w1(array_t11, array_t12))
    array_t2 = np.array([[1,2],[3,4]])
    print("w2",w2(array_t2))
    array_t31 = np.array([[1, 0], [1, 0], [0, 1]])
    array_t32 = np.array([2, 1, 1])
    print("w3",w3(array_t31, array_t32))
    w4(np.array([1,2,3]))

    # for w5, we can use the pre-loaded images
    fig = plt.figure(figsize=(10, 3))
    ax1, ax2, ax3 = fig.subplots(1, 3)
    img = misc.ascent()
    img_180 = w5(img)
    full_img_180 = w5(img)
    ax1.imshow(img, cmap='gray')
    ax1.set_axis_off()
    ax2.imshow(img_180, cmap='gray')
    ax2.set_axis_off()
    ax3.imshow(full_img_180, cmap='gray')
    ax3.set_axis_off()
    fig.set_tight_layout(True)
    plt.show()