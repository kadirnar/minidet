import torch
import torch.nn as nn
from .shufflenetv2 import ShuffleNetV2
from .custom_layers import DetectHead, SPP

class Detector(nn.Module):
    def __init__(self, category_num, load_param):
        super(Detector, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
         
        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P3 = self.upsample(P3)
        P1 = self.avg_pool(P1)
        P = torch.cat((P1, P2, P3), dim=1)

        y = self.SPP(P)

        return self.detect_head(y)
