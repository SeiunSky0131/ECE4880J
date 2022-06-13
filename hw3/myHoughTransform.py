import numpy as np

def Handwrite_HoughTransform(img_threshold, rhostep, thetastep):
    # YOUR CODE HERE
    """
    Inputs:
        img_threshold: input image
        rhostep: the step length of rho
        thetastep: the distance of theta

    Outputs:
        img_houghtrans: accumulator as output
        rhoList: rho axis index
        thetaList: theta axis index
    """
    # First create rholist and thetalist
    rhoList = []
    thetaList = []
    N,M = img_threshold.shape

    # rhomax is the length of the diagnol line
    rhomax = np.sqrt(M * M + N * N)

    # for theta in range(0, 2 * np.pi, thetastep):
    #     thetaList.append(theta)
    theta = 0
    while theta < 2 * np.pi:
        thetaList.append(theta)
        theta += thetastep

    # for rho in range(0, rhomax, rhostep):
    #     rhoList.append(rho)
    rho = 0
    while rho < rhomax:
        rhoList.append(rho)
        rho += rhostep
    
    # initialize the accumulator
    img_houghtrans = np.zeros((len(thetaList), len(rhoList)))

    for y in range (0,N):
        for x in range (0,M):
            print("doing rho theta calculation for x = ",x, " and y = ",y)
            if img_threshold[y][x] != 0:
                rho_calculated_list = x * np.cos(thetaList) + y * np.sin(thetaList)
                for rho_calculated in rho_calculated_list:

                    # use argmin to find the nearest rho in rhoList, it returns the index
                    rho = np.argmin(np.abs(rhoList - rho_calculated)) 
                    img_houghtrans[list(rho_calculated_list).index(rho_calculated)][rho] += 1

    return [img_houghtrans, rhoList, thetaList]