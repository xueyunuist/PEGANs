import torch
from torch import nn, optim, autograd
import numpy as np


class Generator(nn.Module):

    def __init__(self, h_dim=400):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self, h_dim=400):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            # nn.Sigmoid()  # 这里可能要换
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)
