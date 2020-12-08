import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(object):

    def __init__(self, device, ngpu, model):

        self.device, self.ngpu = device, ngpu

        if model == 'SRCNN':
          self.model = SRNET(self.ngpu).to(self.device)
        elif model == 'FSRCNN':
          self.model = FSRCNN(self.ngpu).to(self.device)
        elif model == 'SRResNet':
          self.model = SRResNet(self.ngpu).to(self.device)

        if (self.device.type == 'cuda') and (self.model.ngpu > 0):
            self.model = nn.DataParallel(self.model, list(range(self.model.ngpu)))

        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.model)
        print("The number of parameters: {}".format(num_params))

        self.mse = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)


class SRNET(nn.Module):
    def __init__(self, ngpu):
        super(SRNET, self).__init__()

        self.ngpu = ngpu
        self.model = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9//2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0//2),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=5//2),
          nn.ReLU(),
        )

    def forward(self, input):
        return torch.clamp(self.model(input), min=1e-12, max=1-(1e-12))


class FSRCNN(torch.nn.Module):
    def __init__(self, ngpu, n_channels=3, d=56, s=12, m=2):
        super(FSRCNN, self).__init__()

        self.ngpu = ngpu

        # Feature extraction
        self.extraction = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=d, kernel_size=5, stride=1, padding=5//2),
            nn.PReLU())

        # Shrinking
        self.shrinking = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0//2),
                                         nn.PReLU())

        # Non-linear Mapping
        layers = []
        for _ in range(m):
            layers.append(
                nn.Sequential(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=3//2),
                              nn.PReLU()))
        self.mapping = nn.Sequential(*layers)

        # # Expanding
        self.expanding = nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=1//2),
                                         nn.PReLU())


        # Deconvolution
        self.deconvolution = nn.ConvTranspose2d(in_channels=d, out_channels=n_channels, kernel_size=9, stride=1, padding=9//2,
                                            output_padding=0)

    def forward(self, x):
        out = self.extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)

        return torch.clamp(out, min=1e-12, max=1-(1e-12))


class SRResNet(nn.Module):
    def __init__(self, ngpu, in_channels=3, out_channels=3, B=16):
        super(SRResNet, self).__init__()
        self.ngpu = ngpu
        
        self.input = nn.Sequential(
                        nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False),
                        nn.PReLU(num_parameters=1,init=0.2)
        )
        
        residual = []
        for _ in range(B):
            residual.append(ResidualBlock())
        self.residual = nn.Sequential(*residual)

        self.ConvBN = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64)
        )

        self.shuffle = nn.Sequential(
                        ShuffleBlock(),
                        ShuffleBlock()
        )

        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, x):
        output0 = self.input(x)
        output = self.residual(output0)
        output = self.ConvBN(output)
        output = torch.add(output, output0)
        output = self.shuffle(output)
        output = self.conv(output)
        
        return output



class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.PReLU(num_parameters=1,init=0.2),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64)
        )
        
    def forward(self, x):
        output = self.model(x)
        output = torch.add(output,x) # elementwise sum

        return output 


class ShuffleBlock(nn.Module):
    def __init__(self):
        super(ShuffleBlock, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.PixelShuffle(1),
                        nn.PReLU(num_parameters=1,init=0.2)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output 