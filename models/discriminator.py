import math

import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.IN(x)
        x = self.relu(x)
        return x

class GlobalDiscriminator(nn.Module):
    def __init__(self, pretrainfile=None):
        super(GlobalDiscriminator, self).__init__()
        self.global_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2)
        self.global_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2)
        self.global_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2)
        self.global_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2)
        self.global_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2)
        self.global_conv6 = ConvBnRelu(512, 512, kernel_size=5, stride=2)
        # self.global_conv7 = ConvBnRelu(512, 512, kernel_size=4, stride=4)
        self.global_conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=4)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.init(pretrainfile)
        
    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
        else:
            self.load_state_dict(torch.load(pretrainfile, map_location=lambda storage, loc: storage))
            print ('==> [netD] load self-train weight as pretrain.')        
    
    def forward(self, input):
        x = self.global_conv1(input)
        x = self.global_conv2(x)
        x = self.global_conv3(x)
        x = self.global_conv4(x)
        x = self.global_conv5(x)
        x = self.global_conv6(x)
        x = self.global_conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def calc_loss(self, pred, gt):
        loss = nn.BCEWithLogitsLoss()(pred, gt)
        return loss


class LocalContentDiscriminator(nn.Module):
    def __init__(self):
        super(LocalContentDiscriminator, self).__init__()
        self.local_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2)
        self.local_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2)
        self.local_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2)
        self.local_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2)
        self.local_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2)        
        # self.local_conv6 = ConvBnRelu(512, 512, kernel_size=4, stride=4)
        self.local_conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=4)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, .1)
                m.bias.data.zero_() 

    def forward(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = self.local_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LocalMaskDiscriminator(nn.Module):
    def __init__(self):
        super(LocalMaskDiscriminator, self).__init__()
        self.local_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2)
        self.local_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2)
        self.local_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2)
        self.local_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2)
        self.local_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2)        
        # self.local_conv6 = ConvBnRelu(512, 512, kernel_size=4, stride=4)
        self.local_conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=4)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, .1)
                m.bias.data.zero_()        
    
    def forward(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = self.local_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
