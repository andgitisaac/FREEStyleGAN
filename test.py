import argparse
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
import models.generator as G
import models.mattingNet as M

from utils.ops import adaptive_instance_normalization as adain
from utils.ops import binaryMask2GaussianMask, combine_foreNback_with_mask
from config import *


def concat_compare(img_list):
    x = np.hstack((img_list[0], img_list[1]))
    y = np.hstack((img_list[2], img_list[3]))
    z = np.hstack((img_list[4], img_list[5]))

    return np.vstack((x, y, z))

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def stylizer(net, content, style, soft_mask, cl_mask, sr_mask=None, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    style_feat = net.encode(style)
    content_feat = net.encode(content)

    cl_mask = net.stack_mask_dim(cl_mask, 512, 3)

    if sr_mask is not None:
        sr_mask = net.stack_mask_dim(sr_mask, 512, 3)

    stylized_feat = adain(content_feat, style_feat, cl_mask=cl_mask, sr_mask=sr_mask)
    stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat # stylized content features

    stack_masks = net.stack_mask_dim(soft_mask, 512, 3)
    inverse_stack_masks = torch.ones_like(stack_masks) - stack_masks
    stylized_feat = stylized_feat * stack_masks + style_feat * inverse_stack_masks
    
    decoded = net.decoder(stylized_feat)
    return decoded

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory of input images, \
                        One pair of images should be named as \
                        X_content.jpg, , X_target.jpg, X_mask.jpg, X_naive.jpg \
                        for foreground component, background style image, binary mask, pasted image, respectively')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save the output image(s)')

parser.add_argument('--vgg', type=str, default=PRETRAINED_VGG_PATH)
parser.add_argument('--decoder', type=str, required=True)
parser.add_argument('--mask', type=str, required=True)
parser.add_argument('--decoder_no_local', type=str, default=PRETRAINED_DECODER_PATH)
parser.add_argument('--mask_no_local', type=str, default=PRETRAINED_MATTING_NETWORK_PATH)

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument('--sr', action='store_true',
                    help='output style referenced images')
parser.add_argument('--no_soft', action='store_true',
                    help='deactivate matting network and use binary mask instead of soft mask')
parser.add_argument('--no_local', action='store_true',
                    help='use decoder trained w/o local discriminators')
args = parser.parse_args()

device = torch.device('cuda')

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

default_dir = os.path.join(args.output_dir, 'default')
if not os.path.exists(default_dir):
    os.mkdir(default_dir)

softmask_dir = os.path.join(args.output_dir, 'softmask')
if not os.path.exists(softmask_dir):
    os.mkdir(softmask_dir)

noSoft_dir = os.path.join(args.output_dir, 'noSoft')
if not os.path.exists(noSoft_dir):
    os.mkdir(noSoft_dir)

noLoaclD_dir = os.path.join(args.output_dir, 'noLocalD')
if not os.path.exists(noLoaclD_dir):
    os.mkdir(noLoaclD_dir)

compare_dir = os.path.join(args.output_dir, 'compare')
if not os.path.exists(compare_dir):
    os.mkdir(compare_dir)

### Initializing Networks ###
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

decoder = net.decoder
decoder.load_state_dict(torch.load(args.decoder))

generator = G.Generator(vgg, decoder)
generator.eval()
generator.to(device)

mattingNet = M.MattingNetwork()
mattingNet.load_state_dict(torch.load(args.mask))
mattingNet.eval()
mattingNet.to(device)

print("-----Model Loaded!-----")

image_tf = test_transform()
prefix_names = glob.glob(os.path.join(args.input_dir, '*.jpg'))
prefix_names = set([name.split('/')[-1].split('_')[0] for name in prefix_names])

timer = 0.0

for prefix_name in prefix_names:    
    content = image_tf(Image.open(os.path.join(args.input_dir, str(prefix_name)+'_content.jpg')))
    style = image_tf(Image.open(os.path.join(args.input_dir, str(prefix_name)+'_target.jpg')))
    binary_mask = skimage.io.imread(os.path.join(args.input_dir, str(prefix_name)+'_mask.jpg'))

    # assert (style.size() % 2 == 0 and style.size() / 2 >= 8) # size should be larger than 256*256.
    # assert (style.size() == content.size() == binary_mask.size())

    start_time = time.time()

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)    

    binary_mask = np.expand_dims(binary_mask[:, :, 0], axis=0)
    binary_mask = binary_mask / 255 # quite important...
    gaussian_mask = binaryMask2GaussianMask(binary_mask)

    binary_mask = torch.FloatTensor(binary_mask).unsqueeze(1).to(device)
    gaussian_mask = torch.FloatTensor(gaussian_mask).unsqueeze(1).to(device)
    
    merged_image = combine_foreNback_with_mask(content, style, gaussian_mask)
    content = combine_foreNback_with_mask(content, style, binary_mask) # to make the input content more smooth with style

    ### Non-sr output ###
    ### activate matting network (i.e., w/ soft mask) ###
    with torch.no_grad():
        soft_mask = mattingNet(merged_image, binary_mask)
                
        output = stylizer(generator, content, style, soft_mask, cl_mask=binary_mask, sr_mask=None, alpha=args.alpha)        
        timer += time.time() - start_time

    mask_name = os.path.join(softmask_dir, '{}_softmask.jpg'.format(prefix_name))
    save_image(soft_mask, mask_name)
    
    # paste back w.r.t. binary mask
    output_pasteback = combine_foreNback_with_mask(output, style, binary_mask).cpu()
    output_name = os.path.join(default_dir, '{}_output.jpg'.format(prefix_name))
    save_image(output_pasteback, output_name)

    # paste back w.r.t. soft mask
    output_pasteback_softmask = combine_foreNback_with_mask(output, style, soft_mask).cpu()
    output_name = os.path.join(default_dir, '{}_output_soft.jpg'.format(prefix_name))
    save_image(output_pasteback_softmask, output_name)

    
    ### sr output ###
    # args.alpha == 0: E2STN is doing image matting only.
    if args.alpha != 0 and args.sr:
        with torch.no_grad(): 
            # sr_mask now becomes the same mask as cl_mask
            output_sr = stylizer(generator, content, style, soft_mask, cl_mask=binary_mask, sr_mask=binary_mask, alpha=args.alpha)

        # paste back w.r.t. binary mask
        output_pasteback_sr = combine_foreNback_with_mask(output_sr, style, binary_mask).cpu()
        output_name = os.path.join(default_dir, '{}_output_sr.jpg'.format(prefix_name))
        save_image(output_pasteback_sr, output_name)

        # paste back w.r.t. soft mask
        output_pasteback_softmask_sr = combine_foreNback_with_mask(output_sr, style, soft_mask).cpu()
        output_name = os.path.join(default_dir, '{}_output_soft_sr.jpg'.format(prefix_name))
        save_image(output_pasteback_softmask_sr, output_name)

    ### de-activate matting network (i.e., w/o soft mask) ###
    if args.no_soft:
        with torch.no_grad():
            output_noSoft = stylizer(generator, content, style, binary_mask, cl_mask=binary_mask, sr_mask=None, alpha=args.alpha)
            
            # paste back w.r.t. binary mask
            output_pasteback_noSoft = combine_foreNback_with_mask(output_noSoft, style, binary_mask).cpu()
            output_name = os.path.join(noSoft_dir, '{}_output_noSoft.jpg'.format(prefix_name))
            save_image(output_pasteback_noSoft, output_name)

print('Average time of {} images: {:.4f} secs'.format(len(prefix_names), timer/len(prefix_names)))


### use decoder trained w/o local discriminators ###
if args.no_local:
    ### Initializing Networks ###
    decoder = net.decoder
    decoder.load_state_dict(torch.load(args.decoder_no_local))

    generator = G.Generator(vgg, decoder)
    generator.eval()
    generator.to(device)

    mattingNet = M.MattingNetwork()
    mattingNet.load_state_dict(torch.load(args.mask_no_local))
    mattingNet.eval()
    mattingNet.to(device)
    print("-----Pretrained Model Loaded!-----")

    for prefix_name in prefix_names:    
        content = image_tf(Image.open(os.path.join(args.input_dir, str(prefix_name)+'_content.jpg')))
        style = image_tf(Image.open(os.path.join(args.input_dir, str(prefix_name)+'_target.jpg')))
        binary_mask = skimage.io.imread(os.path.join(args.input_dir, str(prefix_name)+'_mask.jpg'))

        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)    

        binary_mask = np.expand_dims(binary_mask[:, :, 0], axis=0)
        binary_mask = binary_mask / 255 # quite important...
        gaussian_mask = binaryMask2GaussianMask(binary_mask)

        binary_mask = torch.FloatTensor(binary_mask).unsqueeze(1).to(device)
        gaussian_mask = torch.FloatTensor(gaussian_mask).unsqueeze(1).to(device)
        
        merged_image = combine_foreNback_with_mask(content, style, gaussian_mask)
        content = combine_foreNback_with_mask(content, style, binary_mask) # to make the input content more smooth with style

        ### Non-sr output ###
        ### activate matting network (i.e., w/ soft mask) ###
        with torch.no_grad():
            soft_mask = mattingNet(merged_image, binary_mask)                
            output_noLocal = stylizer(generator, content, style, soft_mask, cl_mask=binary_mask, sr_mask=None, alpha=args.alpha)  
        
        # paste back w.r.t. binary mask
        output_pasteback_noLocal = combine_foreNback_with_mask(output_noLocal, style, binary_mask).cpu()
        output_name = os.path.join(noLoaclD_dir, '{}_output_noLocal.jpg'.format(prefix_name))
        save_image(output_pasteback_noLocal, output_name)

print("-----Concatenating-----")
for prefix_name in prefix_names:
    file_str = [os.path.join(default_dir, '{}_output.jpg'.format(prefix_name)),
                os.path.join(default_dir, '{}_output_soft.jpg'.format(prefix_name)),
                os.path.join(default_dir, '{}_output_sr.jpg'.format(prefix_name)),
                os.path.join(default_dir, '{}_output_soft_sr.jpg'.format(prefix_name)),
                os.path.join(noSoft_dir, '{}_output_noSoft.jpg'.format(prefix_name)),
                os.path.join(noLoaclD_dir, '{}_output_noLocal.jpg'.format(prefix_name))]
    img_list = [skimage.io.imread(path) for path in file_str]
    compare_output = concat_compare(img_list)

    skimage.io.imsave(os.path.join(compare_dir, '{}_compare.jpg'.format(prefix_name)), compare_output)