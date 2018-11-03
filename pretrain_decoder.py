import argparse
import os

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import models.network as net
import models.adainNet as adainG
import models.generator as G
import models.discriminator as D

from utils.data import ImageDataset, train_transform
from utils.maskGen import generateBatchMask
from utils.sampler import InfiniteSamplerWrapper
from utils.ops import calc_acc, adjust_learning_rate, combine_foreNback_with_mask, binaryMask2GaussianMask
from utils.ops import soft_label, flip_label, save_model
from config import *

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default=CONTENT_PATH,
                    help='Directory path to content images')
parser.add_argument('--style_dir', type=str, default=STYLE_PATH,
                    help='Directory path to style images')
parser.add_argument('--vgg', type=str, default=PRETRAINED_VGG_PATH)
parser.add_argument('--adain_decoder', type=str, default=PRETRAINED_ADAIN_PATH)
parser.add_argument('--mask', type=str, default=PRETRAINED_MATTING_NETWORK_PATH)

# training options
parser.add_argument('--save_dir', default='./checkpoints/pretrained_decoder/',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs/pretrained_decoder/',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--write_logs_interval', type=int, default=100)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--d_thresh', type=float, default=0.8)
parser.add_argument('--label_flip', type=float, default=0.2)

# weight options
parser.add_argument('--tv_weight', type=float, default=5e-5)
parser.add_argument('--adversarial_weight', type=float, default=1.0)
parser.add_argument('--foreground_weight', type=float, default=1.0)
parser.add_argument('--background_weight', type=float, default=1.0)
parser.add_argument('--style_weight', type=float, default=10.0)
args = parser.parse_args()

cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

writer = SummaryWriter(log_dir=args.log_dir)

### OPs ###
criterion = nn.BCEWithLogitsLoss()

### Initializing Networks ###
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

adain_decoder = adainG.pretrained_decoder
adain_decoder.load_state_dict(torch.load(args.adain_decoder))
adain_stylizer = adainG.Generator(vgg, adain_decoder)
adain_stylizer.eval()
adain_stylizer.to(device)

decoder = net.decoder
generator = G.Generator(vgg, decoder)
generator.train()
generator.to(device)

global_discriminator = D.GlobalDiscriminator()
global_discriminator.train()
global_discriminator.to(device)

parameters_G = list(generator.decoder.parameters())
optimizer_G = torch.optim.Adam(parameters_G, lr=args.lr)
parameters_D_global = list(global_discriminator.parameters())
optimizer_D_global = torch.optim.Adam(parameters_D_global, lr=args.lr)

### Prepare Dataset ###
content_dataset = ImageDataset(args.content_dir, train_transform(cropSize=args.img_size))
style_dataset = ImageDataset(args.style_dir, train_transform(cropSize=args.img_size))

content_loader = iter(DataLoader(dataset=content_dataset,
                            batch_size=args.batch_size,
                            sampler=InfiniteSamplerWrapper(content_dataset),
                            num_workers=args.n_threads))
style_loader = iter(DataLoader(dataset=style_dataset,
                            batch_size=args.batch_size,
                            sampler=InfiniteSamplerWrapper(style_dataset),
                            num_workers=args.n_threads))

for step in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer_G, args_lr=args.lr, args_lr_decay=args.lr_decay, iteration_count=step)
    adjust_learning_rate(optimizer_D_global, args_lr=args.lr, args_lr_decay=args.lr_decay, iteration_count=step)

    # Adversarial Ground-Truth
    real_label, fake_label = soft_label(args.batch_size, device=device)

    original_content_images = next(content_loader).to(device)
    style_images = next(style_loader).to(device)    
    # arbitrary_style_images = next(style_loader).to(device) # use for global D, the 'real' images have to be different from style
    
    masks, _ = generateBatchMask(args.batch_size, args.img_size)
    gaussian_masks = binaryMask2GaussianMask(masks)
    gaussian_masks = torch.FloatTensor(gaussian_masks).unsqueeze(1).to(device)

    with torch.no_grad():
        stylized_content = adain_stylizer(original_content_images, style_images)    
    
    ### Update G with global D ###  
    
    # The input content image is mostly white except the region of foreground
    content_images = combine_foreNback_with_mask(foreground=original_content_images,
                                                background=torch.ones_like(original_content_images),
                                                mask=gaussian_masks)
    loss_dict, stylized_images = generator(content_images, style_images, gaussian_masks)

    real_global = style_images
    # real_global = arbitrary_style_images
    fake_global = stylized_images

    pred_label = global_discriminator(fake_global)

    loss_g = criterion(pred_label, real_label)
    loss_g = args.adversarial_weight * loss_g

    loss_tv = args.tv_weight * loss_dict['tv_loss']
    loss_c_fore = args.foreground_weight * loss_dict['loss_c_fore']
    loss_c_back = args.background_weight * loss_dict['loss_c_back']
    loss_s = args.style_weight * loss_dict['loss_s']    
    loss_G = loss_c_fore + loss_c_back + loss_s + loss_tv + loss_g
    
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()


    ### Update Global D ###
    gt_real, gt_fake = flip_label(real_label, fake_label, args.label_flip, step)

    # Train D with real
    pred_label = global_discriminator(real_global)
    loss_d_real = criterion(pred_label, gt_real)
    acc_d_real_global = calc_acc(pred_label, real_label)

    # Train D with fake
    pred_label = global_discriminator(fake_global.detach())
    loss_d_fake = criterion(pred_label, gt_fake)
    acc_d_fake_global = calc_acc(pred_label, fake_label)

    loss_D_global = (loss_d_real + loss_d_fake) / 2
    loss_D_global = args.adversarial_weight * loss_D_global

    # Update Global D ONLY when D is too weak
    if ((acc_d_fake_global + acc_d_real_global) / 2) < args.d_thresh:
        optimizer_D_global.zero_grad()
        loss_D_global.backward()
        optimizer_D_global.step()

    if (step + 1) % args.write_logs_interval == 0:
        content_images = vutils.make_grid(original_content_images[:4], nrow=4)
        style_images = vutils.make_grid(style_images[:4], nrow=4)
        gaussian_masks = vutils.make_grid(gaussian_masks[:4], nrow=4)
        stylized_images = vutils.make_grid(stylized_images[:4], nrow=4)

        writer.add_image('A_stylized', stylized_images, step + 1)
        writer.add_image('B_content', content_images, step + 1)
        writer.add_image('C_mask', gaussian_masks, step + 1)
        writer.add_image('D_style', style_images, step + 1)

        writer.add_scalars('global_loss', {'loss_g': loss_g.item(), 'loss_D_global': loss_D_global.item()}, step + 1)
        writer.add_scalars('global_D_acc', {'acc_real': acc_d_real_global, 'acc_fake': acc_d_fake_global}, step + 1)
        writer.add_scalars('loss_content', {'loss_foreground': loss_c_fore.item(), 'loss_background': loss_c_back.item()}, step + 1)        
        writer.add_scalars('loss_tv', {'loss_tv': loss_tv.item()}, step + 1)
        writer.add_scalars('loss_style', {'loss_style': loss_s.item()}, step + 1)        

    if (step + 1) % args.save_model_interval == 0 or (step + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        filename = '{:s}/pretrained_decoder_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)

        state_dict = global_discriminator.state_dict()
        filename = '{:s}/global_discriminator_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)

writer.close()
