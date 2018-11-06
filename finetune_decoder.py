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
import models.mattingNet as M

from utils.data import ImageDataset, train_transform
from utils.sampler import InfiniteSamplerWrapper
from utils.maskGen import generateBatchMask
from utils.ops import calc_acc, adjust_learning_rate, binaryMask2GaussianMask, combine_foreNback_with_mask
from utils.ops import get_local_content, soft_label, flip_label, save_model
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
parser.add_argument('--pretrained_decoder', type=str, default=PRETRAINED_DECODER_PATH)

# training options
parser.add_argument('--save_dir', default='./checkpoints/finetuned/',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs/finetuned/',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4)
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
parser.add_argument('--mask_weight', type=float, default=1e3)
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
decoder.load_state_dict(torch.load(args.pretrained_decoder))
generator = G.Generator(vgg, decoder)
generator.train()
generator.to(device)

local_content_discriminator = D.LocalContentDiscriminator()
local_content_discriminator.train()
local_content_discriminator.to(device)

local_mask_discriminator = D.LocalMaskDiscriminator()
local_mask_discriminator.train()
local_mask_discriminator.to(device)

mattingNet = M.MattingNetwork()
mattingNet.load_state_dict(torch.load(args.mask))
mattingNet.train()
mattingNet.to(device)

parameters_G = list(generator.decoder.parameters())
optimizer_G = torch.optim.Adam(parameters_G, lr=args.lr)

parameters_M = list(mattingNet.parameters())
optimizer_M = torch.optim.Adam(parameters_M, lr=args.lr)

parameters_D_local_content = list(local_content_discriminator.parameters())
optimizer_D_local_content = torch.optim.Adam(parameters_D_local_content, lr=args.lr)

parameters_D_local_mask = list(local_mask_discriminator.parameters())
optimizer_D_local_mask = torch.optim.Adam(parameters_D_local_mask, lr=args.lr)

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
    adjust_learning_rate(optimizer_M, args_lr=args.lr, args_lr_decay=args.lr_decay, iteration_count=step)
    adjust_learning_rate(optimizer_D_local_content, args_lr=args.lr, args_lr_decay=args.lr_decay, iteration_count=step)
    adjust_learning_rate(optimizer_D_local_mask, args_lr=args.lr, args_lr_decay=args.lr_decay, iteration_count=step)

    # Adversarial Ground-Truth
    real_label, fake_label = soft_label(args.batch_size, device=device)

    original_content_images = next(content_loader).to(device)
    style_images = next(style_loader).to(device)
    masks, local_coordinate = generateBatchMask(args.batch_size, args.img_size)
    gaussian_masks = binaryMask2GaussianMask(masks)

    masks = torch.FloatTensor(masks).unsqueeze(1).to(device)
    gaussian_masks = torch.FloatTensor(gaussian_masks).unsqueeze(1).to(device)

    with torch.no_grad():
        stylized_content = adain_stylizer(original_content_images, style_images)
        real_local = get_local_content(stylized_content, local_coordinate, args.batch_size).to(device)

    ### Update G with Local Content D (Fix M) ###

    # The input content image is mostly white except the region of foreground
    with torch.no_grad():
        merged_image = combine_foreNback_with_mask(original_content_images, style_images, gaussian_masks)
        soft_masks = mattingNet(merged_image, masks)

        content_images = combine_foreNback_with_mask(foreground=original_content_images,
                                                    background=torch.ones_like(original_content_images),
                                                    mask=soft_masks)
    

    loss_dict, stylized_images = generator(content_images, style_images, soft_masks.detach())

    fake_local = get_local_content(stylized_images, local_coordinate, args.batch_size).to(device)    
    pred_label = local_content_discriminator(fake_local)

    loss_gan_g = criterion(pred_label, real_label)
    loss_gan_g = args.adversarial_weight * loss_gan_g

    loss_tv = args.tv_weight * loss_dict['tv_loss']
    loss_c_fore = args.foreground_weight * loss_dict['loss_c_fore']
    loss_c_back = args.background_weight * loss_dict['loss_c_back']
    loss_s = args.style_weight * loss_dict['loss_s']    
    loss_G = loss_c_fore + loss_c_back + loss_s + loss_tv + loss_gan_g
    
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    ### Update Local Content D ###
    gt_real, gt_fake = flip_label(real_label, fake_label, args.label_flip, step)

    # Train Content D with real
    pred_label = local_content_discriminator(real_local)
    loss_d_content_real = criterion(pred_label, gt_real)
    acc_d_content_real_local = calc_acc(pred_label, real_label, args.batch_size)

    # Train Content D with fake
    pred_label = local_content_discriminator(fake_local.detach())
    loss_d_content_fake = criterion(pred_label, gt_fake)
    acc_d_content_fake_local = calc_acc(pred_label, fake_label, args.batch_size)

    loss_D_content_local = (loss_d_content_real + loss_d_content_fake) / 2
    loss_D_content_local = args.adversarial_weight * loss_D_content_local

    # Update Local Content D ONLY when it is too weak
    if ((acc_d_content_fake_local + acc_d_content_real_local) / 2) < args.d_thresh:
        optimizer_D_local_content.zero_grad()
        loss_D_content_local.backward()
        optimizer_D_local_content.step()

    ### Update M with Local Mask D (Fix G) ###    
    with torch.no_grad():
        merged_image = combine_foreNback_with_mask(original_content_images, style_images, soft_masks)
        _, stylized_images = generator(content_images, style_images, soft_masks)

    refined_soft_masks = mattingNet(merged_image, masks)
    # Re-paste the foreground on the "real" style image
    refined_stylized_images = combine_foreNback_with_mask(foreground=stylized_images.detach(),
                                                    background=style_images,
                                                    mask=refined_soft_masks)
    
    # fake_local = get_local_content(stylized_images, local_coordinate).to(device) #  compare with output directly
    fake_local = get_local_content(refined_stylized_images, local_coordinate, args.batch_size).to(device) # compare with re-paste
        
    # Since G tries to minimize the loss of foreground, the loss of foreground would become zero if 
    # Matting Network outputs a all-0s soft mask. Thus, the gaussian-mask loss works as a regularizer 
    # to prevent the refined soft masks from vanishing. 
    loss_mask = mattingNet.calc_mask_loss(refined_soft_masks, gaussian_masks)
    loss_mask = args.mask_weight * loss_mask

    pred_label = local_mask_discriminator(fake_local)
    loss_gan_m = criterion(pred_label, real_label)
    loss_gan_m = args.adversarial_weight * loss_gan_m

    loss_M = loss_mask + loss_gan_m

    optimizer_M.zero_grad()
    loss_M.backward()
    optimizer_M.step()    

    ### Update Local Mask D ###
    gt_real, gt_fake = flip_label(real_label, fake_label, args.label_flip, step)

    # Train D with real
    pred_label = local_mask_discriminator(real_local)
    loss_d_mask_real = criterion(pred_label, gt_real)
    acc_d_mask_real_local = calc_acc(pred_label, real_label, args.batch_size)

    # Train D with fake
    pred_label = local_mask_discriminator(fake_local.detach())
    loss_d_mask_fake = criterion(pred_label, gt_fake)
    acc_d_mask_fake_local = calc_acc(pred_label, fake_label, args.batch_size)

    loss_D_mask_local = (loss_d_mask_real + loss_d_mask_fake) / 2
    loss_D_mask_local = args.adversarial_weight * loss_D_mask_local

    if ((acc_d_mask_fake_local + acc_d_mask_real_local) / 2) < args.d_thresh:
        optimizer_D_local_mask.zero_grad()
        loss_D_mask_local.backward()
        optimizer_D_local_mask.step()


    if (step + 1) % args.write_logs_interval == 0:
        content_images = vutils.make_grid(original_content_images[:4], nrow=4)
        style_images = vutils.make_grid(style_images[:4], nrow=4)
        refined_soft_masks = vutils.make_grid(refined_soft_masks[:4], nrow=4) 
        refined_stylized_images = vutils.make_grid(refined_stylized_images[:4], nrow=4)

        writer.add_image('A_refined_stylized', refined_stylized_images, step + 1)
        writer.add_image('B_content', content_images, step + 1)
        writer.add_image('C_refined_mask', refined_soft_masks, step + 1)
        writer.add_image('D_style', style_images, step + 1)
            
        writer.add_scalars('local_content_loss', {'loss_g': loss_gan_g.item(), 'loss_D_content_local': loss_D_content_local.item()}, step + 1)
        writer.add_scalars('local_content_D_acc', {'acc_real': acc_d_content_real_local, 'acc_fake': acc_d_content_fake_local}, step + 1)
            
        writer.add_scalars('local_mask_loss', {'loss_m': loss_gan_m.item(), 'loss_D_mask_local': loss_D_mask_local.item()}, step + 1)
        writer.add_scalars('local_mask_D_acc', {'acc_real': acc_d_mask_real_local, 'acc_fake': acc_d_mask_fake_local}, step + 1)
        
        writer.add_scalars('loss_content', {'loss_foreground': loss_c_fore.item(), 'loss_background': loss_c_back.item()}, step + 1)        
        writer.add_scalars('loss_tv', {'loss_tv': loss_tv.item()}, step + 1)
        writer.add_scalars('loss_style', {'loss_style': loss_s.item()}, step + 1)        

    if (step + 1) % args.save_model_interval == 0 or (step + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        filename = '{:s}/finetuned_decoder_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)

        state_dict = mattingNet.state_dict()
        filename = '{:s}/finetuned_mattingNet_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)
        
        state_dict = local_content_discriminator.state_dict()
        filename = '{:s}/local_content_discriminator_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)
        
        state_dict = local_mask_discriminator.state_dict()
        filename = '{:s}/local_mask_discriminator_iter_{:d}.pth'.format(args.save_dir, step + 1)
        save_model(weights=state_dict, filename=filename)
        
writer.close()
