import torch
import torch.nn as nn
from abc import ABC


class MRFNET(nn.Module, ABC):
    def __init__(self, channels):
        super(MRFNET, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))

        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))

        self.conv1_18 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_19 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_20 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_21 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_22 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_23 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_24 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))

        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv1_17 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv1_18o = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                   groups=groups, bias=False)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if 0 <= m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif -clip_b < m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)

        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x2t = self.conv1_15(x1)

        x1 = self.conv1_18(x2t)
        x1 = self.conv1_19(x1)
        x1 = self.conv1_20(x1)
        x1 = self.conv1_21(x1)
        x1 = self.conv1_22(x1)
        x1 = self.conv1_23(x1)
        x1 = self.conv1_24(x1)

        out = torch.cat([self.conv1_16(x1t), self.conv1_17(x2t), self.conv1_18o(x1)], 1)

        out = self.Tanh(out)
        out = self.conv3(out)
        out2 = x - out
        return out2