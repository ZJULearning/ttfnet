import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmdet.models.utils.attention_module import CBAMBlock


class SeparableConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 kernels_per_layer=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels * kernels_per_layer,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer,
                                   out_channels,
                                   kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class TridentConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 group_3x3=3):
        super(TridentConv2d, self).__init__()
        assert group_3x3 >= 1
        self.group_3x3 = group_3x3

        group = group_3x3 + 1  # 1x1 + 3x3-dila-1 + 3x3-dila-2 + ...
        out = int(out_channels * 2 / group)
        self.conv_1x1 = nn.Conv2d(in_channels, out, 1)
        self.weight = nn.Parameter(torch.randn(out, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out))
        nn.init.normal_(self.weight, 0, 0.01)

        self.bottleneck = nn.Conv2d(out * group, out_channels, 1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_fuse = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        out = []
        out.append(self.conv_1x1(x))
        for i in range(1, self.group_3x3 + 1):
            out.append(F.conv2d(x, self.weight, bias=self.bias, padding=i, dilation=i))

        out = torch.cat(out, dim=1)
        out = F.relu(out)

        out = self.bottleneck(out)
        shortcut = self.conv_shortcut(x)
        out = out + shortcut

        out = self.conv_fuse(out)
        out = F.relu(out)
        return out


class ExpConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 neg_x=False):
        super(ExpConv2d, self).__init__()
        self.neg_x = neg_x
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = x.exp()
        if self.neg_x:
            x = -x
        return self.conv(x)


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False,
                 use_prelu=False,
                 use_cbam=False,
                 shortcut_in_shortcut=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)
        self.shortcut_in_shortcut = shortcut_in_shortcut

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                if use_prelu:
                    layers.append(nn.PReLU(out_channels))
                else:
                    layers.append(nn.ReLU(inplace=True))

        # if use_cbam:
        #     layers.append(CBAMBlock(planes=out_channels))

        self.layers = nn.Sequential(*layers)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        y = self.layers(x)
        if self.shortcut_in_shortcut:
            shortcut = self.shortcut(x)
            y = shortcut + y
        return y
