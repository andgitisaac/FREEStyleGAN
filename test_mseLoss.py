import argparse
import os
import time
import glob

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import skimage.io
from collections import defaultdict

import scipy.io

import models.network as net
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

def extract_feats(feat, mask):    
    mask = torch.ne(mask, 0.0) # the Byte Tensor for mask
    feat = torch.masked_select(feat, mask)
    return feat

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--vgg', type=str, default=PRETRAINED_VGG_PATH)
args = parser.parse_args()

device = torch.device('cuda')

### Initializing Networks ###
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.eval()
vgg.to(device)

decoder = net.decoder
# decoder.load_state_dict(torch.load(args.decoder))

generator = G.Generator(vgg, decoder)
generator.eval()
generator.to(device)

print("-----Model Loaded!-----")

image_tf = test_transform()
mse_loss = nn.MSELoss()

methods = ['Ours', 'CNNMRF', 'Adobe', 'AdaIN']
loss_dict = defaultdict(list)

prefix_names = glob.glob(os.path.join(args.input_dir, '*.jpg'))
prefix_names = list(set([int(name.split('/')[-1].split('_')[0]) for name in prefix_names]))
print(len(prefix_names))
for method in methods:

    loss = []
    # loss = 0.0
    for prefix_name in prefix_names:
        content = image_tf(Image.open(os.path.join(args.input_dir, str(prefix_name)+'_content.jpg')))
        binary_mask = skimage.io.imread(os.path.join(args.input_dir, str(prefix_name)+'_mask.jpg'))
        output = image_tf(Image.open(os.path.join(args.output_dir + method, str(prefix_name)+'_output.jpg')))

        content = content.to(device).unsqueeze(0)    
        output = output.to(device).unsqueeze(0)    

        binary_mask = np.expand_dims(binary_mask[:, :, 0], axis=0)
        binary_mask = binary_mask / 255 # quite important...
        binary_mask = torch.FloatTensor(binary_mask).unsqueeze(1).to(device)
        stack_binary_mask = generator.stack_mask_dim(binary_mask, 512, 3)
        
        content_feat = extract_feats(vgg(content), stack_binary_mask)
        output_feat = extract_feats(vgg(output), stack_binary_mask)

        result = mse_loss(output_feat, content_feat).item()
        loss.append(result)
        # loss += mse_loss(output_feat, content_feat)
    loss_dict[method] = loss

    print("{} loss: {:.4f}".format(method, sum(loss)/len(prefix_names)))
scipy.io.savemat('loss', loss_dict)









    