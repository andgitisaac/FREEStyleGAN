import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.mattingNet import MattingNetwork
from utils.sampler import InfiniteSamplerWrapper
from utils.data import ImageDataset, train_transform
from utils.maskGen import generateBatchMask
from utils.ops import binaryMask2GaussianMask, combine_foreNback_with_mask, save_model
from config import CONTENT_PATH, STYLE_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', type=str,
                    default=CONTENT_PATH,
                    help='Directory path to content images')
parser.add_argument('--style_dir', type=str,
                    default=STYLE_PATH,
                    help='Directory path to style images')
parser.add_argument('--save_dir', default='./checkpoints/pretrained_mattingNet/',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs/pretrained_mattingNet/',
                    help='Directory to save the logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--write_logs_interval', type=int, default=100)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

writer = SummaryWriter(log_dir=args.log_dir)

mattingNetwork = MattingNetwork()
mattingNetwork.train()
mattingNetwork.to(device)

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

parameters = list(mattingNetwork.parameters())
optimizer = torch.optim.Adam(parameters, lr=args.lr)
for step in tqdm(range(args.max_iter)):
    content_images = next(content_loader).to(device)
    style_images = next(style_loader).to(device)  

    masks, _ = generateBatchMask(args.batch_size, args.img_size)
    gaussian_masks = binaryMask2GaussianMask(masks)

    masks = torch.FloatTensor(masks).unsqueeze(1).to(device)
    gaussian_masks = torch.FloatTensor(gaussian_masks).unsqueeze(1).to(device)

    merged_image = combine_foreNback_with_mask(content_images, style_images, gaussian_masks)

    softMask = mattingNetwork(merged_image, masks)
    loss = mattingNetwork.calc_mask_loss(softMask, gaussian_masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % args.write_logs_interval == 0:
        original_masks = vutils.make_grid(masks[:4], nrow=4)
        target = vutils.make_grid(gaussian_masks[:4], nrow=4)
        smoothed_masks = vutils.make_grid(softMask[:4], nrow=4)
        merged_image = vutils.make_grid(merged_image[:4], nrow=4)
        
        writer.add_scalar('loss', loss.item(), step + 1)        
        writer.add_image('A_Original', original_masks, step + 1)
        writer.add_image('C_Target', target, step + 1)
        writer.add_image('B_Smoothed', smoothed_masks, step + 1)
        writer.add_image('D_Input_images', merged_image, step + 1)
        

    if (step + 1) % args.save_model_interval == 0 or (step + 1) == args.max_iter:
        state_dict = mattingNetwork.state_dict()
        filename = '{:s}/pretrained_mattingNet_iter_{:d}.pth'.format(args.save_dir,
                                                                    step + 1)
        save_model(weights=state_dict, filename=filename)
        
writer.close()