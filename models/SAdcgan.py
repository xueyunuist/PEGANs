import torch
import torch.nn as nn
import torch.nn.init as init
from models.attention import SelfAttention, ContextBlock
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 stride=1, padding=1, self_attention=False, GC=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        # self.activation = activation(negative_slope=0.1, inplace=True)
        self.self_attention = self_attention
        if self_attention:
            self.attention = SelfAttention(out_channel)
        self.GC = GC
        if GC:
            self.GC = ContextBlock(out_channel, out_channel // 8, mode='add', pooling_type='att')
        # self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, input):
        out = self.conv(input)
        # out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=True)
        if self.self_attention:
            out = self.attention(out)
        if self.GC:
            out = self.GC(out)
        return out


class ConvTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 stride=1, padding=1,
                 activation=F.relu, self_attention=False, GC=False):
        super().__init__()

        self.convTrans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
        self.activation = activation
        self.self_attention = self_attention
        self.GC = GC
        if self_attention:
            self.attention = SelfAttention(out_channel)
        if GC:
            self.GC = ContextBlock(out_channel, out_channel // 8, mode='add', pooling_type='att')
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, input):
        out = self.convTrans(input)
        out = self.bn(out)
        out = self.activation(out)
        if self.self_attention:
            out = self.attention(out)
        if self.GC:
            out = self.GC(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            ConvTransBlock(512, 256, kernel_size=4, stride=2, padding=1, activation=F.relu, self_attention=False),
            ConvTransBlock(256, 128, kernel_size=4, stride=2, padding=1, activation=F.relu, self_attention=False),
            ConvTransBlock(128, 64, kernel_size=4, stride=2, padding=1, activation=F.relu, self_attention=True,
                           GC=False),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(3, 64, 3, 1, 1, self_attention=False),

            # nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(64, 128, 4, 2, 1, self_attention=True, GC=False),
            # M / 2
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, self_attention=False, GC=False),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, self_attention=False, GC=False),
            # M / 4
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, self_attention=False, GC=False),
            ConvBlock(256, 512, kernel_size=4, stride=2, padding=1, self_attention=False, GC=False),
            # M / 8
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1, inplace=True)
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, self_attention=False)
        )

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def rescale_weight(self, min_norm=1.0, max_norm=1.33):
        a = 1.0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    w_norm = m.weight.norm(p=2)
                    print(m, w_norm)
                    w_norm = max(w_norm, min_norm)
                    w_norm = min(w_norm, max_norm)
                    a = a * w_norm
                    m.weight.data.div_(w_norm)
                    m.bias.data.div_(a)

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=6)


class Discriminator32(Discriminator):
    def __init__(self, *args):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self, *args):
        super().__init__(M=48)
