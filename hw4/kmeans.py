from turtle import shape
import numpy as np
import matplotlib.pyplot as plt

data = np.load('kmeans_array2.npy')

# Visualize data
# plt.scatter(data[:, 0], data[:, 1], c='c', s = 30,marker='o')
# plt.show()

### Start your K-means ###
# print(data.shape)

# analyze the data points, find x range and y range
x_min = data[:, 0].min()
x_max = data[:, 0].max()
y_min = data[:, 1].min()
y_max = data[:, 1].max()

# print(x_min, x_max, y_min, y_max)
# define K and iteration numbers
K = 5
iter_num = 16

# define the color to plot
colormap = {}
colormap[0] = 'red'
colormap[1] = 'green'
colormap[2] = 'blue'
colormap[3] = 'yellow'
colormap[4] = 'orange'

# initialize the membership matrix and distance matrix
membership_matrix = np.zeros((data.shape[0], K))
distance_matrix = np.zeros((data.shape[0], K))

# randomly initailize the center
centers = []
for ii in range(0, K):
    x = np.random.random() * (x_max - x_min) + x_min
    y = np.random.random() * (y_max - y_min) + y_min
    centers.append((x,y))
    # print((x,y))


error_list = []
for j in range(0, iter_num):
    print("Iteration: ", j, "\n")

    # calculate the distance matrix
    for m in range(0, data.shape[0]):
        for n in range(0, K):
            distance_matrix[m][n] = (data[m][0] - centers[n][0]) * (data[m][0] - centers[n][0]) + (data[m][1] - centers[n][1]) * (data[m][1] - centers[n][1])

    # update the membership matrix
    for m in range(0, data.shape[0]):
        closest_center = np.argmin(distance_matrix[m])
        for n in range(0, K):
            membership_matrix[m][n] = 1 if n == closest_center else 0

    # update the centers
    x_sum_for_centers = np.zeros(K)
    y_sum_for_centers = np.zeros(K)
    num_for_centers = np.zeros(K)
    for m in range(0, data.shape[0]):
        n = np.argmax(membership_matrix[m]) # find the center for this point
        x_sum_for_centers[n] += data[m][0]
        y_sum_for_centers[n] += data[m][1]
        num_for_centers[n] += 1

    for i in range(0, K):
        centers[i] = (x_sum_for_centers[i]/num_for_centers[i], y_sum_for_centers[i]/num_for_centers[i])
            
    error = 0
    for m in range(0, data.shape[0]):
        n = np.argmax(membership_matrix[m])

        # plot the data
        plt.scatter(data[m][0], data[m][1], color = colormap[n])

        error += (data[m][0] - centers[n][0]) * (data[m][0] - centers[n][0]) + (data[m][1] - centers[n][1]) * (data[m][1] - centers[n][1])
    
    # plot the centers
    for i in range(0, K):
        plt.scatter(centers[i][0], centers[i][1], color = colormap[i], marker = '*')
    if j == 15:
        plt.savefig('K=5, iter%d_set2.png'%j, dpi = 480)
    plt.close()
    print("Objective value is: ", error)
    error_list.append(error)

plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],error_list)
plt.savefig('error_each_iter_set2_nouse.png', dpi = 480)


