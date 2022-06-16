from contextlib import suppress
from typing import Tuple, Union 

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

from myConvolution  import  myConv2D


def GaussianKernel(sigma:float=1.):
    """Generate a gaussian kernel with the given standard deviation.

    The kernel size is decided by 2*ceil(3*sigma) + 1
    """
    # TODO: calculate the kernel size and initialize
    size = int(2 * np.ceil(3 * sigma) + 1)

    # TODO:  generate the gaussian kernel
    # initialize the kernel
    kernel = np.zeros((size, size))

    # The center should have (x,y) = (0,0), assume size is odd
    m = size/2
    for i in range(0, size):
        for j in range(0, size):
            x = i - m
            y = j - m
            kernel[i][j] = (1/(2 * np.pi * sigma * sigma)) * np.exp(-(x * x + y * y)/(2 * sigma * sigma))

    return kernel


def SobelFilter(img:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # the Sobel operators on x-direction and y-direction
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
    Ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], np.float32)

    # TODO: calculate the intensity gradient
    Gx = myConv2D(img, Kx, stride = 1, padding = 1)
    Gy = myConv2D(img, Ky, stride = 1, padding = 1)
    G = np.sqrt(Gx**2 + Gy**2)
    # add small noise to Gx's 0
    for i in range(0,Gx.shape[0]):
        for j in range(0, Gx.shape[1]):
            if Gx[i][j] == 0:
                Gx[i][j] = 0.001
    theta = np.arctan(Gy/Gx)

    return G, theta


def NMS(img:np.ndarray, angles:np.ndarray) -> np.ndarray:
    """Process the image with non-max suppression algorithm
    """
    # map the gradient angle to the closest of 4 cases, where the line is sloped at 
    #   almost 0 degree, 45 degree, 90 degree, and 135 degree
    if img.shape != angles.shape:
        print("gradient and its angle have different size!")
        return
    else:
        N,M = img.shape
        # padding the image to enable NMS at edge
        img = np.pad(img,((1,1),(1,1)), mode = "constant")

        suppressed = np.zeros(img.shape)
        for i in range(1,N + 1):
            for j in range(1,M + 1):

                # case 1, degree = 0
                if (angles[i - 1][j - 1] >= -np.pi/8) and (angles[i - 1][j - 1] < np.pi/8):
                    if (img[i][j] >= img[i][j - 1]) and (img[i][j] >= img[i][j + 1]):
                        suppressed[i - 1][j - 1] = img[i][j]
                    else:
                        suppressed[i - 1][j - 1] = 0
                
                # case 2, degree = 45
                elif (angles[i - 1][j - 1] >= np.pi/8) and (angles[i - 1][j - 1] < np.pi*3/8):
                    if (img[i][j] >= img[i + 1][j - 1]) and (img[i][j] >= img[i - 1][j + 1]):
                        suppressed[i - 1][j - 1] = img[i][j]
                    else:
                        suppressed[i - 1][j - 1] = 0

                # case 3, degree = 90
                elif (angles[i - 1][j - 1] >= np.pi*3/8) or (angles[i - 1][j - 1] < -np.pi*3/8):
                    if (img[i][j] >= img[i - 1][j]) and (img[i][j] >= img[i + 1][j]):
                        suppressed[i - 1][j - 1] = img[i][j]
                    else:
                        suppressed[i - 1][j - 1] = 0
                
                # case 4. degree = 135
                elif (angles[i - 1][j - 1] >= -np.pi*3/8) and (angles[i - 1][j - 1] < -np.pi/8):
                    if (img[i][j] >= img[i - 1][j - 1]) and (img[i][j] >= img[i + 1][j + 1]):
                        suppressed[i - 1][j - 1] = img[i][j]
                    else:
                        suppressed[i - 1][j - 1] = 0

        return suppressed


def myCanny(
    img:np.ndarray, 
    sigma:Union[float, Tuple[float,float]]=1.,
    threshold:Tuple[int,int]=(100,150)) -> np.ndarray:
    """Apply Canny algorithm to detect the edge in an image.

    Returns: The edge detection result whose size is the same as the input image.
    """
    # denoise the image by a convolution with the gaussian filter
    #   TODO: implement the Gaussian kernel generator
    gaussian_kernel = GaussianKernel(sigma)
    padding = np.floor(np.array(gaussian_kernel.shape)/2)
    denoised = myConv2D(img, gaussian_kernel, padding=(int(padding[0]), int(padding[1])))
    plt.imsave("./zebra_denoised.jpg", denoised, cmap="Greys_r")

    # find the intensity gradient of the image
    #   TODO: implement the Sobel filter
    gradient, angles = SobelFilter(denoised)
    plt.imsave("./zebra_gradient.jpg", gradient, cmap="Greys_r")

    # find the edge candidates by non-max suppression
    #   TODO: implement the non-max suppression function
    nms = NMS(gradient, angles)
    plt.imsave("./zebra_nms.jpg", nms, cmap="Greys_r")

    # TODO: determine the potential edges by the hysteresis threshold
    N,M = nms.shape
    # Again we need padding
    nms = np.pad(nms, ((1,1),(1,1)), mode = "constant")
    output = np.zeros(nms.shape)

    for i in range (1, N + 1):
        for j in range(1, M + 1):
            
            # if strong
            if nms[i][j] >= threshold[1]:
                output[i - 1][j - 1] = nms[i][j]

            # if weaker than weak
            elif nms[i][j] < threshold[0]:
                output[i - 1][j - 1] = 0

            # if weak
            elif (nms[i][j] >= threshold[0]) and (nms[i][j] < threshold[1]):
                strong_flag = False
                for x in range (i - 1, i + 2):
                    for y in range (j - 1, j + 2):
                        if nms[x][y] >= threshold[1]:
                            strong_flag = True
                            break
                        else:
                            continue
                if strong_flag:
                    output[i - 1][j - 1] = nms[i][j]
                else:
                    output[i - 1][j - 1] = 0

    plt.imsave("./zebra_edge.jpg", output, cmap="Greys_r")
    return output

# following code is testing code
if __name__ == "__main__":
    # print(GaussianKernel(1))
    image = mpimg.imread('hw3_zebra.jpg')
    image = image.astype('float')
    myCanny(image)
