import torch
import torch.nn as nn

from utils.ops import adaptive_instance_normalization as adain
from utils.ops import calc_mean_std

class Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()

        # encoder
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        # decoder
        self.decoder = decoder

        # ops
        self.mse_loss = nn.MSELoss()
        self.calc_tv_loss = lambda x: torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) + \
                                torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        self.downsample = nn.Sequential(nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True))

        # arguments
        self.args = [[64, 0], [128, 1], [256, 2], [512, 3]] # (# of channels, # of downsampling)

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


    def encode_with_intermediate(self, input_image):
        '''Extract features of input image from relu1_1, relu2_1, relu3_1, relu4_1.
            
        Params:
        input_image - tensors of image

        Returns [output_1, output_2, output_3, output_4],
        where output_X denotes the features at reluX_1.
        '''
        results = [input_image]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


    def encode(self, input_image):
        '''Extract features of input image from relu4_1.'''
        for i in range(4):
            input_image = getattr(self, 'enc_{:d}'.format(i + 1))(input_image)
        return input_image

    def stack_mask_dim(self, mask, n_channel, downsample_times):
        '''Stack mask n_channel times and downsample it downsample_times times.'''
        stack_masks = torch.squeeze(mask, dim=1)                    # bs x 1 x 256 x 256 -> bs x 256 x 256
        for _ in range(downsample_times):
            stack_masks = self.downsample(stack_masks)              # bs x 256 x 256 -> bs x (256/2^downsample_times) x (256/2^downsample_times)
        stack_masks = torch.stack([stack_masks]*n_channel, dim=1)   # bs x 32 x (256/2^downsample_times) x (256/2^downsample_times)
        return stack_masks
    
    def linear_combine_content_and_style(self, content, style, mask):
        '''Embed features of content image into features of style image'''
        mask = self.stack_mask_dim(mask, 512, 3)
        inverse_mask = torch.ones_like(mask) - mask
        # Use style as background, content as foreground
        output = content * mask + \
                        style * inverse_mask 
        return output    
   
    def calc_weighted_content_loss(self, input, target, weight):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return (weight * (input - target) ** 2).mean()

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, mask, alpha=1.0):
        assert 0 <= alpha <= 1

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        stylized_feat = adain(content_feats[-1], style_feats[-1])
        stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feats[-1] # stylized content feature

        combined_stylized_feat = self.linear_combine_content_and_style(stylized_feat, style_feats[-1], mask)

        stylized_output = self.decoder(combined_stylized_feat)
        stylized_output_feats = self.encode_with_intermediate(stylized_output)

        # Calculate multi-scale content loss
        loss_c_fore, loss_c_back = 0.0, 0.0
        for i in range(0, 4):
            mask_resized = self.stack_mask_dim(mask, self.args[i][0], self.args[i][1])
            inverse_mask_resized = torch.ones_like(mask_resized) - mask_resized
            loss_c_fore += self.calc_weighted_content_loss(stylized_output_feats[i], content_feats[i], mask_resized)
            loss_c_back += self.calc_weighted_content_loss(stylized_output_feats[i], style_feats[i], inverse_mask_resized)

        # Calculate style loss
        loss_s = 0.0
        for i in range(4):
            loss_s += self.calc_style_loss(stylized_output_feats[i], style_feats[i])
        
        # Calculate tv loss
        tv_loss = self.calc_tv_loss(stylized_output)

        loss = {'loss_s': loss_s,
                'loss_c_fore': loss_c_fore,
                'loss_c_back': loss_c_back,
                'tv_loss': tv_loss}

        return loss, stylized_output