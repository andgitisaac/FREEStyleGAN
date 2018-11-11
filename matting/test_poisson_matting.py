from __future__ import division

import glob, os
import numpy as np
import scipy.ndimage
import cv2
from numba import jit


@jit
def computeAlphaJit(alpha, b, unknown):
    h, w = unknown.shape
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4  * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - b[i,j])
        n +=1
    return alphaNew


def poisson_matte(gray_img, trimap):
    h, w = gray_img.shape
    fg = trimap == 255
    bg = trimap == 0
    unknown = True ^ np.logical_or(fg,bg)
    fg_img = gray_img*fg
    bg_img = gray_img*bg
    alphaEstimate = fg + 0.5 * unknown

    approx_bg = cv2.inpaint(bg_img.astype(np.uint8),(unknown+fg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(fg)).astype(np.float32)
    approx_fg = cv2.inpaint(fg_img.astype(np.uint8),(unknown+bg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(bg)).astype(np.float32)

    # Smooth F - B image
    approx_diff = approx_fg - approx_bg
    approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)

    dy, dx = np.gradient(gray_img)
    d2y, _ = np.gradient(dy/approx_diff)
    _, d2x = np.gradient(dx/approx_diff)
    b = d2y + d2x

    alpha = computeAlphaJit(alphaEstimate, b, unknown)
    
    alpha = np.minimum(np.maximum(alpha,0),1).reshape(h,w)
    return alpha

# Load in image
def main():    
    input_dir = "/home/andgitisaac/dataset/input/"
    output_dir = "/home/andgitisaac/dataset/matting_output/poisson/"
    prefix_names = glob.glob('{}*.jpg'.format(input_dir))
    prefix_names = set([prefix_name.split('/')[-1].split('_')[0] for prefix_name in prefix_names])
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, prefix_name in enumerate(prefix_names):
        print('Processing FILE: {} ({}/{})...'.format(prefix_name, i+1, len(prefix_names)))

        naive = scipy.misc.imread("/home/andgitisaac/dataset/input/{}_naive.jpg".format(prefix_name))
        gray_naive = scipy.misc.imread("/home/andgitisaac/dataset/input/{}_naive.jpg".format(prefix_name), flatten='True')
        style = scipy.misc.imread("/home/andgitisaac/dataset/input/{}_target.jpg".format(prefix_name))
        trimap = scipy.misc.imread("/home/andgitisaac/dataset/input/{}_trimap.jpg".format(prefix_name))

        alpha = poisson_matte(gray_naive, trimap[:, :, 0])
        
        alpha = np.dstack((alpha, alpha, alpha))
                
        scipy.misc.imsave(os.path.join(output_dir, '{}_alpha.jpg'.format(prefix_name)), alpha)
        inverse = np.ones(alpha.shape) - alpha
        output = naive * alpha + style * inverse
        
        scipy.misc.imsave(os.path.join(output_dir, '{}_matting.jpg'.format(prefix_name)), output)


if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()