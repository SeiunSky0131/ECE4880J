from cgi import test
from typing import Tuple, Union
import numpy as np
import matplotlib.image as mpimg
from matplotlib.image import imsave

def myConv2D(
    img:np.ndarray, kernel:np.ndarray, 
    stride:Union[int, Tuple[int,int]] = 1, 
    padding:Union[int, Tuple[int, int]]=0) -> np.ndarray:
    """Convolve two 2-dimensional arrays.

    Args:
        img (np.ndarray): A grayscale image.
        kernel (np.ndarray): A odd-sized 2d convolution kernel.
        stride (int or tuple, optional): The parameter to control the movement of the kernel. 
            An integer input k will be automatically converted to a tuple (k, k). Defaults to 1.
        padding (int or tuple, optional): The parameter to control the amount of padding to the image. 
            An integer input k will be automatically converted to a tuple (k, k). Defaults to 0.

    Returns (np.ndarray): The processed image.
    """
    # TODO: check the datatype of stride and padding
    # hint: isinstance()

    if isinstance(stride, int):
        # stride is int
        stride = (stride, stride)

    if isinstance(padding, int):
        # padding is int
        padding = (padding,padding)


    # TODO: define the size of the output and initialize

    output_0 = int(np.floor((img.shape[0] + 2 * padding[0] - kernel.shape[0] + stride[0]) / stride[0]))
    output_1 = int(np.floor((img.shape[1] + 2 * padding[1] - kernel.shape[1] + stride[1]) / stride[1]))
    output = np.zeros((output_0,output_1))

    # TODO: add padding to the image
    # use default value 0 to pad

    img = np.pad(img, ((padding[0], padding[0]),(padding[1],padding[1])), mode = 'constant')
    # print(img)

    # TODO: implement your 2d convolution
    for i in range(0, output_0):
        for j in range(0, output_1):
            # if i == 0 and j == 0:
                # print(i * stride[0] + padding[0] - int(kernel.shape[0]/2)
                # ,i * stride[0] + padding[0] + int(kernel.shape[0]/2) + 1 
                # ,j * stride[1] + padding[1] - int(kernel.shape[1]/2)
                # ,j * stride[1] + padding[1] + int(kernel.shape[1]/2) + 1
                # ,img[i * stride[0] + padding[0] - int(kernel.shape[0]/2) : i * stride[0] + padding[0] + int(kernel.shape[0]/2) + 1, j * stride[1] + padding[1] - int(kernel.shape[1]/2) : j * stride[1] + padding[1] + int(kernel.shape[1]/2) + 1])
            output[i][j] = np.sum(kernel * img[i * stride[0] : i * stride[0] + kernel.shape[0], j * stride[1]  : j * stride[1] + kernel.shape[1]])

    return output


# Following code is testing code
if __name__ == "__main__":
    img = mpimg.imread('lena.jpg')
    img = img.astype('float')

    test_mat = np.array([[1,1,1,1,1],[1,2,2,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,1,1,1,1]])
    id_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
    avg_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]/9)
    print("identical kernel: \n", myConv2D(test_mat, id_kernel, stride = 1, padding = (1,1)))
    print("average_kernel: \n", myConv2D(test_mat, avg_kernel,stride = 1, padding = 1))
    print("identical kernel with 2 stride: \n", myConv2D(test_mat, id_kernel, stride = 2, padding = 1))
    
    Guass_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]/16)
    filtered_img = myConv2D(img, Guass_kernel, stride = 1, padding = 1)
    imsave("Conv_lena.jpg",filtered_img,cmap = 'gray')