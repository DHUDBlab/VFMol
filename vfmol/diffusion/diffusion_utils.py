import torch
from torch.nn import functional as F
# from torch.distributions.categorical import Categorical
import numpy as np

from vfmol.utils import PlaceHolder


def assert_correctly_masked(variable, node_mask):
    # 检查变量是否正确地根据 node_mask 做了 mask（屏蔽），也就是没有“泄露”非法的位置的信息
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_discrete_features(probX, probE, node_mask, mask=False):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    从给定的概率分布中采样离散特征，包括节点特征 probX、边特征 probE 和（可能）全局特征 proby
    使用了多项式分布来从每个类别的概率分布中进行采样。
    :param probX: bs, n, dx_out        node features   probX 中的每个元素表示该节点属于某个类别的概率
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]  # 虚拟节点设均匀分布

    # Flatten the probability tensor to sample with multinomial  为了方便使用 multinomial 方法进行采样
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1, replacement=True)  # (bs * n, 1)  每个元素表示该节点被分配到的类别（类别的索引）
    # X_t = Categorical(probs=probX).sample()  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))  # 哪些节点对是无效的（即不应该存在边）
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)  # (bs, n, n)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1, replacement=True).reshape(bs, n, n)  # (bs, n, n)
    # E_t = Categorical(probs=probE).sample().reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)  # 变为上三角矩阵
    E_t = E_t + torch.transpose(E_t, 1, 2)

    if mask:  # 仅保留有效节点
        X_t = X_t * node_mask
        E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process
    一个离散扩散模型中用于从极限分布（limit distribution）采样噪声"""
    bs, n_max = node_mask.shape
    # 节点类别的概率分布（例如 (dx_out,)）
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)  # bs,n,dx
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    # 对每个节点/边采样一个类别（Multinomial）
    U_X = (  # flatten 是把 (bs, n, dx_out) 展平成 (bs*n, dx_out)
        x_limit.flatten(end_dim=-2).multinomial(1, replacement=True).reshape(bs, n_max)
    )
    U_E = (
        e_limit.flatten(end_dim=-2)
        .multinomial(1, replacement=True)
        .reshape(bs, n_max, n_max)
    )
    # U_X = Categorical(probs=x_limit.flatten(end_dim=-2)).sample().reshape(bs, n_max)
    # U_E = Categorical(probs=e_limit.flatten(end_dim=-2)).sample().reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    # one-hot 编码回去
    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)  # 上三角（不含对角线）的索引对 (i, j)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1  # 创建掩码（全 0），只在 上三角位置设置为 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)
