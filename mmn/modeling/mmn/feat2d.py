import torch
from torch import nn


class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d


class SparseConv(nn.Module):
    def __init__(self, pooling_counts, N, hidden_size):
        super(SparseConv, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
            map2d[:, :, i, j] = x
        return map2d


def build_feat2d(cfg):
    pooling_counts = cfg.MODEL.MMN.FEAT2D.POOLING_COUNTS  # [15,8,8] anet, [15] charades
    num_clips = cfg.MODEL.MMN.NUM_CLIPS  # 64 anet, 16 charades
    hidden_size = cfg.MODEL.MMN.FEATPOOL.HIDDEN_SIZE  # 512
    if cfg.MODEL.MMN.FEAT2D.NAME == "conv":
        return SparseConv(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.MMN.FEAT2D.NAME == "pool":
        return SparseMaxPool(pooling_counts, num_clips)
    else:
        raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.MMN.FEAT2D.NAME)
