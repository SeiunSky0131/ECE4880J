from filecmp import cmp
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def get_DCTMatrix(dim):
    mat = np.ones((dim,dim))
    for i in range (0,dim):
        for j in range(0,dim):
            mat[i][j] = np.sqrt(2) * np.cos((2 * j - 1) * (i - 1) * np.pi / (2 * dim))
    return mat

def DCT_transform(img):
    """
    Inputs:
    - img: An 2-D numpy array of shape (H,W)

    Returns:
    - img_dct: the dct transformation result, a 2-D numpy array of shape (H, W)

    Hint: implement the dct transformation basis matrix
    """
    H,W = img.shape

    img_dct = (1 / np.sqrt(H)) * (get_DCTMatrix(H) @ img @ (get_DCTMatrix(H).T))
    return img_dct

def iDCT_transform(img_dct):
    """
    Inputs:
    - img_dct: An 2-D numpy array of shape (H,W)

    Returns:
    - img_recover: recoverd image, a 2-D numpy array of shape (H, W)

    Hint: use the same dct transformation basis matrix but do the reverse operation
    """
    H,W = img_dct.shape

    img_recover = (1/np.sqrt(H)) * ((get_DCTMatrix(H).T) @ img_dct @ get_DCTMatrix(H))
    return img_recover


def main():
    #############################################
    ############ Global compression #############
    #############################################
    image = mpimg.imread('lena.jpg')
    image = image.astype('float')
    H,W = image.shape

    image_dct = DCT_transform(image)

    ### Visualize the log map of dct (image_dct_log) here ###
    image_dct_log = np.log(abs(image_dct))

    ### Compress the dct result here by preserving 1/4 data (H/2,W/2) in image_dct and set others to zero here ###
    image_dct_compress = image_dct * np.pad(np.ones((int(H / 2), int(W / 2))),(0, W - int(W / 2)))
    image_recover = iDCT_transform(image_dct)
    image_compress_recover = iDCT_transform(image_dct_compress)
    # print(image_compress_recover.shape)
    # print(image_recover)
    # print(image_compress_recover)
    imsave("frequency_domain.jpg", image_dct_log, cmap = 'gray')
    imsave("original_lena.jpg", image_recover, cmap = 'gray')
    imsave("compressed_lena_1_4.jpg", image_compress_recover, cmap = 'gray')

    image_dct_compress_16 = image_dct * np.pad(np.ones((int(H / 4), int(W / 4))), (0, W - int(W / 4)))
    image_compress_recover_16 = iDCT_transform(image_dct_compress_16)
    imsave("compressed_lena_1_16.jpg", image_compress_recover_16, cmap = 'gray')

    #############################################
    ########## Blockwise compression ############
    #############################################
    image = mpimg.imread('lena.jpg')
    image = image.astype('float')

    patches_num_h = int(H/8)
    patches_num_w = int(W/8)

    img_recover = np.zeros(image.shape)
    image_recover_compress = np.zeros(image.shape)
    image_dct_log = np.zeros(image.shape)

    for i in range(patches_num_h):
        for j in range(patches_num_w):
            ### divide the image into 8x8 pixel image patches here ###
            patch = image[8 * i : 8 * i + 8, 8 * j : 8 * j + 8]
            patch_dct = DCT_transform(patch)

            patch_dct_log = np.log(abs(patch_dct))

            ### Compress the dct result here by preserving 1/4 data (H/2,W/2) in image_dct and set others to zero here
            patch_dct_compress = patch_dct * np.pad(np.ones((int(8 / 2), int(8 / 2))),(0, 8 - int(8 / 2)))

            patch_recover = iDCT_transform(patch_dct)
            patch_compress_recover = iDCT_transform(patch_dct_compress)

            ### put patches together here
            img_recover[8 * i : 8 * i + 8, 8 * j : 8 * j + 8] = patch_recover
            image_recover_compress[8 * i : 8 * i + 8, 8 * j : 8 * j + 8] = patch_compress_recover
            image_dct_log[8 * i : 8 * i + 8, 8 * j : 8 * j + 8] = patch_dct_log

    imsave("frequency_domain_local.jpg", image_dct_log, cmap = 'gray')
    imsave("original_lena_local.jpg", img_recover, cmap = 'gray')
    imsave("compressed_lena_local.jpg", image_recover_compress, cmap = 'gray')

if __name__ == "__main__":
    main()


 
