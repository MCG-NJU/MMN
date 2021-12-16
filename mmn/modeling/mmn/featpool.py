import torch
from torch import nn


class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T
        return self.pool(self.conv(x).relu())

def build_featpool(cfg):
    input_size = cfg.MODEL.MMN.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.MMN.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.MMN.FEATPOOL.KERNEL_SIZE  # 4 for anet, 2 for tacos, 16 for charades
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.MMN.NUM_CLIPS
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)
