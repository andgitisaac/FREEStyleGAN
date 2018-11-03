import random

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn

### Style Transfer Calculation ###
def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    
    if mask is not None:
        mask = torch.ne(mask, 0.0) # the Byte Tensor for mask
        feat = torch.masked_select(feat, mask)

    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat, cl_mask=None, sr_mask=None):
    # cl_mask: mask to indicate the location of foregound content
    # sr_mask: mask to indicate the region of the referenced style
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, mask=sr_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=cl_mask)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


### Model Related ###
def save_model(weights, filename):
    for key in weights.keys():
        weights[key] = weights[key].to(torch.device('cpu'))
        torch.save(weights, filename)


### Mask Related ###
def binaryMask2GaussianMask(mask, sigma=5):
    smoothed = np.zeros(mask.shape, dtype='float32')
    for i, m in enumerate(mask):
        smoothed[i] = gaussian_filter(m, sigma=sigma)
    return smoothed

def stack_mask(mask, out_channels):
    ''' transform (bs, C, H, W) -> (bs, out_channels, H, W) '''     
    stacked = torch.squeeze(mask, dim=1)
    stacked = torch.stack([stacked]*out_channels, dim=1)
    return stacked

def combine_foreNback_with_mask(foreground, background, mask, out_channels=3):
    mask = stack_mask(mask, out_channels)
    inversed_mask = torch.ones_like(mask) - mask
    output = foreground * mask + background * inversed_mask
    return output


#### Discriminator Related ###
def get_local_content(img, coordinate, batch_size, local_size=128):
    output = torch.zeros((batch_size, 3, local_size, local_size))
    for i, (content, coord) in enumerate(zip(img, coordinate)):
        # coord = (U, L, B, R)
        output[i] = content[:, coord[0]:coord[2], coord[1]:coord[3]]
    return output

def soft_label(batch_size, device):
    real_label = torch.Tensor(batch_size, 1).uniform_(0.7, 1.0).to(device)
    fake_label = torch.Tensor(batch_size, 1).uniform_(0.0, 0.3).to(device)
    return real_label, fake_label

def flip_label(real_label, fake_label, threshold, step):
    ''' 
    There's a chance(threshold) that the label will be flipped every other step,
    in order to weaken the discriminators'''

    gt_real = real_label
    gt_fake = fake_label

    if (step + 1) % 2 == 0:
        if random.random() < threshold:
            gt_real = fake_label
            gt_fake = real_label
    return gt_real, gt_fake 


### Optimizers Related ###
def adjust_learning_rate(optimizer, args_lr, args_lr_decay, iteration_count):
    """Imitating the original implementation"""
    lr = args_lr / (1.0 + args_lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr