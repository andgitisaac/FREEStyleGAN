import os 
import glob
import numpy as np
import skimage.io
from scipy.ndimage import grey_dilation

input_dir = "/home/andgitisaac/dataset/input/"
output_dir = "/home/andgitisaac/dataset/input/"
paths = glob.glob(input_dir + '*_mask.jpg')


for path in paths:
    prefix_name = path.split('/')[-1].split('_')[0]
    mask = skimage.io.imread(path)
    mask = mask[:, :, 0]
    mask = ((mask > 0)*255).astype('uint8')

    dilated_mask = grey_dilation(mask, size=(11)).astype(mask.dtype)
    dilated_mask = ((dilated_mask > 0)*255).astype('uint8')
    dilated_mask = (((dilated_mask - mask) > 0)*127).astype('uint8')

    trimap = (dilated_mask + mask).astype('uint8')
    trimap = np.dstack((trimap, trimap, trimap))

    output_name = os.path.join(output_dir, '{}_trimap.jpg'.format(prefix_name))
    skimage.io.imsave(output_name, trimap)