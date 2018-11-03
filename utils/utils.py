import os
import random
import glob
from PIL import Image
import skimage.transform
import skimage.io
import numpy as np

def get_file_paths(dir, dataType):
    dir = '{}*.jpg'.format(dir) if dataType == 'background' else '{}*.png'.format(dir)
    path = glob.glob(dir)
    return sorted(path)

def get_one_image(path, crop_size=None):
    img = skimage.io.imread(path)
    if len(img.shape) < 3:
        # To make sure image is M*N*3
        img = np.dstack((img,img,img))

    if crop_size is not None:
        img = random_crop(img, crop_size)    
    return img

def random_crop(img, cropTo):
    h, w, _ = img.shape
    crop_size = min(h, w) if min(h, w) < cropTo else cropTo

    offset_h = random.randint(0, h-crop_size)
    offset_w = random.randint(0, w-crop_size)
    img = img[offset_h:offset_h+crop_size, offset_w:offset_w+crop_size, :]

    if img.shape[0] < cropTo:
        img = skimage.transform.resize(img, (cropTo, cropTo), mode='reflect')
        img = (img * 255).astype(np.uint8)
    
    return img

def random_paste(fore, back):
    limit_ratio = random.uniform(0.01, 0.04)

    back_h, back_w, _ = back.shape
    fore_h, fore_w, _ = fore.shape

    alpha = (fore[:, : ,-1] != 0)
    resize_ratio = np.sqrt((back_h * back_w * limit_ratio) / np.sum(alpha, dtype=np.float32))
    new_fore_size = (int(fore_w*resize_ratio), int(fore_h*resize_ratio))

    fore = Image.fromarray(fore)
    back = Image.fromarray(back)
    # alpha = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha = Image.fromarray((alpha).astype(np.uint8))
    mask = Image.new('L', (back_w, back_h))

    fore = fore.resize(new_fore_size, resample=Image.BILINEAR)
    alpha = alpha.resize(new_fore_size, resample=Image.BILINEAR)

    offset_h = random.randint(0, back_h-fore.size[1])
    offset_w = random.randint(0, back_w-fore.size[0])

    back.paste(fore, (offset_w, offset_h), fore)
    back = np.asarray(back, dtype=np.uint8)
    mask.paste(alpha, (offset_w, offset_h))
    mask = np.asarray(mask, dtype=np.uint8)

    return back, mask