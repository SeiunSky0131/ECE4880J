from array import array
from tkinter import W
import numpy as np


def w1(X):
    """
    Input:
    - X: A numpy array

    Returns:
    - A matrix Y such that Y[i, j] = X[i, j] * 10 + 100

    Hint: Trust that numpy will do the right thing
    """
    Y = X*10 + 100 # element-wise operation
    return Y


def w2(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N, N)
    - Y: A numpy array of shape (N, N)

    Returns:
    A numpy array Z such that Z[i, j] = X[i, j] + 10 * Y[i, j]

    Hint: Trust that numpy will do the right thing
    """
    Z = X + 10*Y
    return Z


def w3(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N, N)
    - Y: A numpy array of shape (N, N)

    Returns:
    A numpy array Z such that Z[i, j] = X[i, j] * Y[i, j] - 10

    Hint: By analogy to +, * will do the same thing
    """
    Z = X*Y -10
    return Z


def w4(X, Y):
    """
    Inputs:
    - X: Numpy array of shape (N, N)
    - Y: Numpy array of shape (N, N)

    Returns:
    A numpy array giving the matrix product X times Y

    Hint: Not the same as *!
    """
    return X@Y


def w5(X):
    """
    Inputs:
    - X: A numpy array of shape (N, N) of floating point numbers

    Returns:
    A numpy array with the same data as M, but cast to 32-bit integers

    Hint: astype
    """
    return X.astype(np.int32)


def w6(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N,) of integers
    - Y: A numpy array of shape (N,) of integers

    Returns:
    A numpy array Z such that Z[i] = float(X[i]) / float(Y[i])

    Hint: Dividing two integers is not the same as dividing two floats
    """
    Z = X.astype(float)/Y.astype(float)
    return Z


def w7(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    - A numpy array Y of shape (N * M, 1) containing the entries of X in row
      order. That is, X[i, j] = Y[i * M + j, 0]

    Hint:
    1) np.reshape
    2) You can specify an unknown dimension as -1
    """
    Z = X.reshape(-1,)
    return Z


def w8(N):
    """
    Inputs:
    - N: An integer

    Returns:
    A numpy array of shape (N, 2N)

    Hint: The error "data type not understood" means you probably called
    np.ones or np.zeros with two arguments, instead of a tuple for the shape
    """
    return np.zeros((N,2*N))


def w9(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M) where each entry is between 0 and 1

    Returns:
    A numpy array Y where Y[i, j] = True if X[i, j] > 0.5

    Hint: Trust python to do the right thing
    """
    Y = X > 0.5 # return the matrix under given condition
    return Y


def w10(N):
    """
    Inputs:
    - N: An integer

    Returns:
    A numpy array X of shape (N,) such that X[i] = i

    Hint: np.arange
    """
    return np.arange(N)


def w11(A, v):
    """
    Inputs:
    - A: A numpy array of shape (N, F)
    - v: A numpy array of shape (F, 1)

    Returns:
    Numpy array of shape (N, 1) giving the matrix-vector product Av
    """
    return A@v


def w12(A, v):
    """
    Inputs:
    - A: A numpy array of shape (N, N), of full rank
    - v: A numpy array of shape (N, 1)

    Returns:
    Numpy array of shape (N, 1) giving the matrix-vector product of the inverse
    of A and v: A^-1 v
    """
    return np.linalg.inv(A)@v


def w13(u, v):
    """
    Inputs:
    - u: A numpy array of shape (N, 1)
    - v: A numpy array of shape (N, 1)

    Returns:
    The inner product u^T v

    Hint: .T
    """
    return (u.T)@v


def w14(v):
    """
    Inputs:
    - v: A numpy array of shape (N, 1)

    Returns:
    The L2 norm of v: norm = (sum_i^N v[i]^2)^(1/2)
    You MAY NOT use np.linalg.norm
    """
    return np.sqrt(np.sum(v**2))


def w15(X, i):
    """
    Inputs:
    - X: A numpy array of shape (N, M)
    - i: An integer in the range 0 <= i < N

    Returns:
    Numpy array of shape (M,) giving the ith row of X
    """
    return X[i]


def w16(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    The sum of all entries in X

    Hint: np.sum
    """
    return np.sum(X)


def w17(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array S of shape (N,) where S[i] is the sum of row i of X

    Hint: np.sum has an optional "axis" argument
    """
    return np.sum(X, axis = 1)


def w18(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array S of shape (M,) where S[j] is the sum of column j of X

    Hint: Same as above
    """
    return np.sum(X, axis = 0)


def w19(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array S of shape (N, 1) where S[i, 0] is the sum of row i of X

    Hint: np.sum has an optional "keepdims" argument
    """
    return np.sum(X, axis = 1, keepdims = True)


def w20(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array S of shape (N, 1) where S[i] is the L2 norm of row i of X
    """
    return np.sqrt(np.sum(X**2, axis = 1, keepdims =  True))

# The following code is testing functions
# if __name__ == "__main__":
#     array_t = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     print("w1:",w1(array_t))
#     array_t21 = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
#     array_t22 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     print("w2: ",w2(array_t21, array_t22))
#     print("w3:",w3(array_t21, array_t22))
#     print("w4:",w4(array_t21,array_t22))
#     array_t5 = np.array([[1.1,2.2,3.3],[4.4,5.5,6.6],[7.7,8.8,9.9]])
#     print("w5:",w5(array_t5))
#     array_t61 = np.array([[1,2,3,4,5,6,7,8,9]])
#     array_t62 = np.array([[9,8,7,6,5,4,3,2,1]])
#     print("w6",w6(array_t61,array_t62),np.shape(array_t61), np.shape(array_t62))
#     print("w7:",w7(array_t21), np.shape(array_t21))
#     print("w8:",w8(4))
#     array_t9 = np.array([[0.1,0.5,0.6],[0.7,0.9,0.5],[0.2,0.1,0.3]])
#     print("w9",w9(array_t9))
#     print("w10",w10(10))
#     array_t111 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     array_t112 = np.array([[1],[2],[3]])
#     print("w11",w11(array_t111,array_t112),np.shape(w11(array_t111,array_t112)))
#     array_t121 = np.array([[1,2],[3,4]])
#     array_t122 = np.array([[5],[6]])
#     print("w12",w12(array_t121, array_t122), np.shape(w12(array_t121, array_t122)))
#     array_t131 = np.array([1,2,4,5])
#     array_t132 = np.array([2,3,4,5])
#     print("w13",w13(array_t131,array_t132))
#     array_t14 = np.array([[1],[2],[3],[4]])
#     print("w14",w14(array_t14), np.shape(array_t14))
#     array_t15 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     print("w15",w15(array_t15,1), np.shape(w15(array_t15,1)))
#     array_t16 = array_t15
#     print("w16",w16(array_t16))
#     array_t17 = array_t16
#     print("w17",w17(array_t17))
#     array_t18 = array_t17
#     print("w18",w18(array_t18))
#     array_t19 = array_t18
#     print("w19",w19(array_t19))
#     array_t20 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     print("w20",w20(array_t20))
