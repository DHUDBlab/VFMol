import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features
            将节点特征投影为图级特征"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """X: bs, n, dx."""
        m = X.mean(dim=1)     # 所有节点的均值特征，shape: (bs, dx)
        mi = X.min(dim=1)[0]  # 所有节点的最小值特征，shape: (bs, dx)
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))  # (bs, 4*dx)
        out = self.lin(z)  # (bs, dy)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features. 将边特征投影为图级特征"""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))  # # 所有边的均值特征，shape: (bs, de)
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    # 每个节点 i 对其他节点 j 的注意力权重（在 dim=2 上） 做 softmax，
    # 但要用 softmax_mask 掩盖掉某些位置（例如 padding 节点、非法边、不存在的连接）
    if mask.sum() == 0:  # 没有有效值，不做 softmax，直接返回
        return x
    x_masked = x.clone()
    # mask: 布尔掩码，True 表示有效元素，False 表示要屏蔽（变成 -∞）
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
