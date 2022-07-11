import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

def pca_basic():
    ### Data
    c1 = np.array([[2,2,2],[1,2,3]])
    c2 = np.array([[4,5,6],[3,3,4]])
    c = np.concatenate((c1,c2),axis=1)

    ### Calculate w and w0 here

    # find the samples mean
    s_mean = np.average(c, axis = 1).reshape(-1,1)

    centered_c = c - s_mean
    # print(centered_c)

    # calculate covarience matrix sigma
    sigma = np.cov(centered_c)
    # print(sigma)

    # eigen decomposition of sigma
    e_values, e_vectors = eig(sigma)
    # print("e_values: ", e_values, "\n", "e_vectors:",e_vectors)

    # find the maximum eigen value's index
    max_e_value_index = 0
    max_e_value = 0
    for i in range(0, len(e_values)):
        if e_values[i] > max_e_value:
            max_e_value_index = i
            max_e_value = e_values[i]

    max_e_vector = e_vectors[:,max_e_value_index].reshape(-1,1) # transform to column vector 
    w = max_e_vector
    w0 = -(w.T)@s_mean

    print('w is:', w)
    print('w0 is:', w0)

    # plot the samples
    fig = plt.figure()
    plt.scatter(c[0],c[1])

    # plot the line
    x = np.linspace(0, 8, num = 50)
    y = (-w0-w[0]*x)/w[1]

    plt.plot(x,y.reshape(-1))
    plt.grid()
    # plt.show()
    plt.savefig("q1_1.png", dpi = 480)

    ### Plot the reconstructed points here
    # print(centered_c.shape)
    # print(max_e_vector.shape)
    # print(max_e_vector.shape)
    projection = max_e_vector.T@centered_c
    # print(projection)
    reconstruction = max_e_vector@projection + s_mean

    plt.scatter(c[0],c[1], color = 'blue', label = 'Original Points')
    plt.scatter(reconstruction[0], reconstruction[1], color = 'red', label = 'Reconstructed Points')
    plt.legend()
    plt.savefig("p1_2.png", dpi = 480)


    ### Calculate MSE here
    sum = np.sum((c - reconstruction)**2)

    MSE = sum/c.shape[1]

    print('MSE is:', MSE)
    ### Calculate the Fisher Ratio here
    projection = projection.reshape(-1)
    m1 = np.mean(projection[0:3])
    m2 = np.mean(projection[3:6])
    delta1 = np.var(projection[0:3])
    delta2 = np.var(projection[3:6])
    FR = (m1 - m2) * (m1 - m2) / (delta1 * delta1 + delta2 * delta2)

    print('Fisher Ratio is:', FR)
    return w,w0,MSE,FR


pca_basic()


