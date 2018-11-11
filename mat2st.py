import os
import time
import glob
import random

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import skimage.io

import models.network as net
import models.adainNet as adainG
import models.generator as G
import models.mattingNet as M

from utils.ops import adaptive_instance_normalization as adain
from utils.ops import binaryMask2GaussianMask, combine_foreNback_with_mask
from config import *

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def stylizer(net, content, style, cl_mask, sr_mask=None, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    style_feat = net.encode(style)
    content_feat = net.encode(content)

    cl_mask = net.stack_mask_dim(cl_mask, 512, 3)

    if sr_mask is not None:
        sr_mask = net.stack_mask_dim(sr_mask, 512, 3)

    stylized_feat = adain(content_feat, style_feat, cl_mask=cl_mask, sr_mask=sr_mask)
    stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat # stylized content features
    
    decoded = net.decoder(stylized_feat)
    return decoded

device = torch.device('cuda')
### Initializing Networks ###
vgg = net.vgg
vgg.load_state_dict(torch.load(PRETRAINED_VGG_PATH))
vgg = nn.Sequential(*list(vgg.children())[:31])

adain_decoder = adainG.pretrained_decoder
adain_decoder.load_state_dict(torch.load(PRETRAINED_ADAIN_PATH))
adain_stylizer = adainG.Generator(vgg, adain_decoder)
adain_stylizer.eval()
adain_stylizer.to(device)

print("-----Model Loaded!-----")

methods = ['bayesian', 'poisson']
for method in methods:
    print('========== {} matting =========='.format(method))

    input_dir = "/home/andgitisaac/dataset/input/"
    matting_dir = "/home/andgitisaac/dataset/matting_output/{}/".format(method)
    output_dir = "/home/andgitisaac/dataset/mat2st/{}/".format(method)
    ours_dir = "/home/andgitisaac/FREEStyleGAN/output/default/default/"
    prefix_names = glob.glob('{}*.jpg'.format(input_dir))
    prefix_names = set([prefix_name.split('/')[-1].split('_')[0] for prefix_name in prefix_names])
    

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    compare_dir = os.path.join(output_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.mkdir(compare_dir)

    image_tf = test_transform()
    prefix_names = glob.glob(os.path.join(input_dir, '*.jpg'))
    prefix_names = set([name.split('/')[-1].split('_')[0] for name in prefix_names])

    timer = 0.0

    for prefix_name in prefix_names:
        # content = image_tf(Image.open(os.path.join(input_dir, str(prefix_name)+'_content.jpg')))
        style = image_tf(Image.open(os.path.join(input_dir, str(prefix_name)+'_target.jpg')))
        binary_mask = skimage.io.imread(os.path.join(input_dir, str(prefix_name)+'_mask.jpg'))

        matting = image_tf(Image.open(os.path.join(matting_dir, str(prefix_name)+'_matting.jpg')))
        alpha_mask = skimage.io.imread(os.path.join(matting_dir, str(prefix_name)+'_alpha.jpg'))

        # content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        matting = matting.to(device).unsqueeze(0)   

        binary_mask = np.expand_dims(binary_mask[:, :, 0], axis=0) / 255
        alpha_mask = np.expand_dims(alpha_mask[:, :, 0], axis=0) / 255

        binary_mask = torch.FloatTensor(binary_mask).unsqueeze(1).to(device)
        alpha_mask = torch.FloatTensor(alpha_mask).unsqueeze(1).to(device)

        all_ones = torch.ones(binary_mask.size(), dtype=torch.float32).to(device)
        

        ## Global Style Transfer ###
        with torch.no_grad():                
            output = stylizer(adain_stylizer, matting, style, cl_mask=all_ones, sr_mask=None)

        output_pasteback = combine_foreNback_with_mask(output, style, alpha_mask).cpu()
        output_name = os.path.join(output_dir, '{}_output_global.jpg'.format(prefix_name))
        save_image(output_pasteback, output_name)

        ## Local Style Transfer ###
        with torch.no_grad():                
            output = stylizer(adain_stylizer, matting, style, cl_mask=alpha_mask, sr_mask=None)

        output_pasteback = combine_foreNback_with_mask(output, style, alpha_mask).cpu()
        output_name = os.path.join(output_dir, '{}_output_local.jpg'.format(prefix_name))
        save_image(output_pasteback, output_name)

        file_str = [os.path.join(matting_dir, str(prefix_name)+'_matting.jpg'),
                    os.path.join(ours_dir, str(prefix_name)+'_output.jpg'),
                    os.path.join(ours_dir, str(prefix_name)+'_output_sr.jpg'),
                    os.path.join(output_dir, '{}_output_global.jpg'.format(prefix_name)),
                    os.path.join(output_dir, '{}_output_local.jpg'.format(prefix_name))]
        img_list = [skimage.io.imread(p) for p in file_str]
        output_name = os.path.join(compare_dir, '{}_compare.jpg'.format(prefix_name))
        skimage.io.imsave(output_name, np.hstack(img_list))




