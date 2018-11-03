import torch
import torch.nn as nn

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.padding = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3))
        self.relu = nn.ReLU(inplace=True)
                
    def forward(self, x):
        residual = x
        out = self.padding(x)
        out = self.conv(out)
        out = self.relu(out)
        out = self.padding(out)
        out = self.conv(out)
        out += residual
        out = self.relu(out)
        return out


class MattingNetwork(nn.Module):
    def __init__(self):
        super(MattingNetwork, self).__init__()
        self.padding = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(4, 64, (3, 3)) # mask(# of Channel=1) + content(# of Channel=3)
        self.conv2 = nn.Conv2d(67, 64, (3, 3))
        self.conv3 = nn.Conv2d(64, 1, (3, 3))
        self.resLayer = self.make_layer(67)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def make_layer(self, n_channels):
        layers = []
        for _ in range(3):
            layers.append(ResidualBlock(n_channels))
        return nn.Sequential(*layers)

    def forward(self, input_content, input_mask):
        output = self.padding(torch.cat((input_mask, input_content), dim=1))
        output = self.conv1(output)

        output = self.resLayer(torch.cat((output, input_content), dim=1))       

        output = self.padding(output)
        output = self.conv2(output)

        output = self.padding(output)
        output = self.conv3(output)
        output = self.sigmoid(output)

        return output
    
    def calc_mask_loss(self, pred, target):
        return self.loss(pred, target)