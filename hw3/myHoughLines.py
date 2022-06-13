import numpy as np

def find_local_maxima(H, neighbor_size=5):
    """
    This is a predefined function tool for you to use to find the local maxinum point in a matrix
    Inputs:
    - H: An 2-D numpy array of shape (H,W)

    Returns:
    - peaks: A 2-D numpy array of shape (H,W), where peaks have original value in H, otherwise it is 0

    """
    H_copy = np.copy(H)
    ssize = int((neighbor_size-1)/2)
    peaks = np.zeros(H_copy.shape)
    h, w = H_copy.shape
    for y in range(ssize, h-ssize):
        for x in range(ssize, w-ssize):
            val = H_copy[y, x]
            if val > 0:
                neighborhood = np.copy(H_copy[y-ssize:y+ssize+1, x-ssize:x+ssize+1])
                neighborhood[ssize, ssize] = 0
                if val > np.max(neighborhood):
                    peaks[y, x] = val
    return peaks

def Handwrite_HoughLines(Im, num_lines):
    neighbor_size = 5
    peaks = find_local_maxima(Im, neighbor_size)
    # YOUR CODE HERE
    # initialize a list of tuples containing (value, index of theta, index of rho)
    candidates = []

    # intalize the rhos and thetas
    rhos = []
    thetas = []

    for i in range(0, peaks.shape[0]):
        for j in range(0, peaks.shape[1]):
            if peaks[i][j] != 0:
                candidates.append((peaks[i][j], i, j))

    # sort the tuples by value
    candidates.sort(key = lambda x: x[0], reverse = True)

    for t in range(0, num_lines):
        rhos.append(candidates[t][2])
        thetas.append(candidates[t][1])
    return rhos, thetas
