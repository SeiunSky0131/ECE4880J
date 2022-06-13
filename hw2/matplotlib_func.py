from email.mime import image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def w1():
    """
    Draw a line chart and save as a jpg file

    Hint: plt.subplot(), ax.plot()
    """
    data_x = [1, 2, 3, 4]
    data_y = [1, 4, 2, 3]
    ax = plt.subplot(1,1,1) # 1 row, 1 column, at index 1
    ax.plot(data_x, data_y)
    plt.savefig("w1.jpg",dpi = 480) # save as a jpg file with 480dpi
    plt.close() # remember to close the figure window. Otherwise new plot will be added on it
    return None

def w2():
    """
    Draw a scatter chart and save as a jpg file

    Hint: Use ax.scatter()
    """
    data_x = [1, 2, 3, 4]
    data_y = [1, 4, 2, 3]
    ax = plt.subplot(1,1,1)
    ax.scatter(data_x, data_y)
    plt.savefig("w2.jpg",dpi = 480)
    plt.close()
    return None

def w3():
    """
    Draw a chart with multiple lines of x,x^2,x^3 and save as a jpg file

    Hint: Trust plt.plot()
    """
    x = np.arange(0., 5., 0.2)
    plt.plot(x,x)
    plt.plot(x,x**2)
    plt.plot(x,x**3)
    plt.legend(["x","x^2","x^3"])
    plt.savefig("w3.jpg",dpi = 480)
    plt.close()
    return None

def w4():
    """
    Draw a histogram chart values-names and save as a jpg file
    
    Hint: plt.bar()
    """
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]
    plt.bar(x = names, height = values)
    plt.savefig("w4.jpg", dpi = 480)
    return None

def w5():
    """
    Read and save an image with B channel (R G B channel) as a jpg file

    Hint: matplotlib.image.imread()
    """
    img_path = 'img.jpg'
    img_array = matplotlib.image.imread(img_path)
    img_B = np.apply_along_axis(func1d = lambda x: [0,0,x[2]], axis = 2, arr = img_array)
    print(np.shape(img_B))
    matplotlib.image.imsave("w5.jpg",img_B, dpi = 480)
    return None

# The following code is textcases
if __name__ == "__main__":
    w1()
    w2()
    w3()
    w4()
    w5()