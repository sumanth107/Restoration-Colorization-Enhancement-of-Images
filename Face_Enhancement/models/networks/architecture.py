

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from Face_Enhancement.models.networks.normalization import SPADE


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.opt = opt
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        
        if "spectral" in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        
        spade_config_str = opt.norm_G.replace("spectral", "")
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, opt)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, opt)

    
    def forward(self, x, seg, degraded_image):
        x_s = self.shortcut(x, seg, degraded_image)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, degraded_image)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, degraded_image)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, degraded_image):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, degraded_image))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)



class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out



class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class SPADEResnetBlock_non_spade(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.opt = opt
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        
        if "spectral" in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

       
        spade_config_str = opt.norm_G.replace("spectral", "")
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc, opt)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc, opt)

    
    def forward(self, x, seg, degraded_image):
        x_s = self.shortcut(x, seg, degraded_image)

        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, degraded_image):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
