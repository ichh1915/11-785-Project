import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1//2, bias=False),
                        nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        output = self.model(x)
        
        return output 


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=1//2, bias=False),
                        nn.InstanceNorm2d(channels),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=1//2, bias=False),
                        nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        output = self.model(x) * 2
        output = torch.add(output,x) # elementwise sum

        return output 


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        self.model = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=1//2, output_padding=2//2)
        
    def forward(self, x):
        output = self.model(x)

        return output 


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=1, stride=2, padding=1//2, bias=False),
                        nn.PixelShuffle(2)
        )

    def forward(self, x):
        output = self.model(x)

        return output 


class ResamplerNet(nn.Module):
    def __init__(self, ngpu, bicubic):
        super(ResamplerNet, self).__init__()
        self.ngpu = ngpu
        dim1 = 32
        dim2 = dim1 * 2
        dim3 = dim2 * 2
        self.layer1 = nn.Sequential(
                        ConvBlock(3, dim1),
                        DownsampleBlock(dim1, dim2)
                        #, DownsampleBlock(dim2, dim2)
        )

        res = [ResBlock(dim2)] * 2
        self.res = nn.Sequential(*res)
        # self.ds = DownsampleBlock(dim2, dim3)

        self.kernel_layer = nn.Sequential(
                              # ConvBlock(dim3, dim3),
                              # ConvBlock(dim3, dim3),
                              ConvBlock(dim2, dim2),
                              UpsampleBlock(dim2)
        )

        res2 = [ResBlock(dim2)] * 2
        self.res2 = nn.Sequential(*res2)

        stride = 1 if not bicubic else 2
        self.conv = nn.Conv2d(in_channels=dim2, out_channels=3, kernel_size=1, stride=stride, padding=1//2)

    def forward(self, x):
        output0 = self.layer1(x)
        output = self.res(output0)
        output = torch.add(output, output0)
        # output = self.ds(output)
        output = self.kernel_layer(output)
        output = self.res2(output)
        output = self.conv(output)


        return output 
