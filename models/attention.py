import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = nn.Conv1d(in_channel, in_channel // 8, 1)

        self.key = nn.Conv1d(in_channel, in_channel // 8, 1)

        self.value = nn.Conv1d(in_channel, in_channel // 8, 1)

        self.hx = nn.Conv1d(in_channel // 8, in_channel, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = self.hx(attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class ContextBlock(nn.Module):
    def __init__(self, in_channel, planes, mode, pooling_type='att'):
        super(ContextBlock, self).__init__()
        self.in_channel = in_channel
        self.planes = planes

        self.pooling_type = pooling_type
        self.gamma = nn.Parameter(torch.tensor(0.0))

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channel, 1, kernel_size=1)  # W_k
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if mode == 'add':
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.in_channel, self.planes, kernel_size=1),
                                                  nn.LayerNorm([self.planes, 1, 1]),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(self.planes, self.in_channel, kernel_size=1))
        else:
            self.channel_add_conv = None
        if mode == 'mul':
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.in_channel, self.planes, kernel_size=1),
                                                  nn.LayerNorm([self.planes, 1, 1]),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(self.planes, self.in_channel, kernel_size=1))
        else:
            self.channel_mul_conv = None
        # self.reset_parameters()

        # def reset_parameters(self):
        #     if self.pooling_type == 'att':
        #         kaiming_init(self.conv_mask, mode='fan_in')
        #         self.conv_mask.inited = True
        #
        #     if self.channel_add_conv is not None:
        #         last_zero_init(self.channel_add_conv)
        #     if self.channel_mul_conv is not None:
        #         last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        if self.pooling_type == 'att':
            batch, channel, height, width = x.size()
            input_x = x
            # [N,C,H*W]
            input_x = input_x.view(batch, channel, height * width)
            # [N,1,C,H*W]
            input_x = input_x.unsqueeze(1)
            # [N,1,H,W]
            context_mask = self.conv_mask(x)
            # [N,1,H*W]
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            # [N,1,H*W,1]
            context_mask = context_mask.unsqueeze(-1)
            # [N,1,C,1]
            context = torch.matmul(input_x, context_mask)
            # [N,C,1,1]
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N,C,1,1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term * self.gamma

        return out
