import torch
import torch.nn as nn
import copy

from vfmol.metrics.abstract_metrics import (
    CrossEntropyMetric,
    KLDMetric,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def js_divergence(q, p, eps=1e-10):
    # JS散度 between batch of q and a single p.
    q = q + eps  # 防止 log(0)
    p = p.unsqueeze(0) + eps  # broadcast to (1, d)

    m = 0.5 * (q + p)

    js = 0.5 * (q * (q.log() - m.log())).sum(dim=1) + \
         0.5 * (p * (p.log() - m.log())).sum(dim=1)
    return js.mean()


def kl_divergence(q, p, eps=1e-10):
    q = q + eps  # 防止 log(0)
    p = p.unsqueeze(0) + eps  # broadcast to (1, d)
    return (q * (q.log() - p.log())).sum(dim=1).mean()


def compute_mmd(q, p, kernel='rbf', sigma=1.0):
    # 计算 softmax 后概率分布之间的 MMD 损失。
    """
    Args:
        q: Tensor of shape (batch_size, d) —— encoder 输出（q(z|x)）
        p: Tensor of shape (d,) —— 先验分布 p(z)
        kernel: 核函数类型，'rbf' 或 'linear'
        sigma: RBF 核的宽度
    Returns:
        标量 MMD 损失
    """
    p = p.unsqueeze(0).expand(q.size(0), -1)  # broadcast (bs, d)

    def _rbf(x, y, sigma):
        # 高斯核函数
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (bs, bs, d)
        return torch.exp(-((diff ** 2).sum(-1)) / (2 * sigma ** 2))  # (bs, bs)

    def _linear(x, y):
        return torch.matmul(x, y.T)  # (bs, bs)

    if kernel == 'rbf':
        K_qq = _rbf(q, q, sigma)
        K_pp = _rbf(p, p, sigma)
        K_qp = _rbf(q, p, sigma)
    elif kernel == 'linear':
        K_qq = _linear(q, q)
        K_pp = _linear(p, p)
        K_qp = _linear(q, p)
    else:
        raise ValueError("Only 'rbf' and 'linear' kernels are supported.")

    # MMD^2 = E[qq] + E[pp] - 2 * E[qp]
    mmd = K_qq.mean() + K_pp.mean() - 2 * K_qp.mean()
    return mmd


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    # 训练时用于计算离散标签（分类）损失的模块，
    # 主要处理节点 (X)、边 (E) 以及图级属性 (y) 的多类交叉熵损失或 KLD（如果指定的话），
    # 并支持标签平滑（Label Smoothing）、类别权重（Class Weight）和日志记录（用于训练可视化）

    def __init__(self, lambda_train, label_smoothing, class_weight, kld=False):
        super().__init__()
        self.lambda_train = lambda_train  # 控制 loss_E 和 loss_y 的加权因子，[5,0]
        if not kld:  # kld=False
            self.node_loss = CrossEntropyMetric(label_smoothing, class_weight.X)
            self.edge_loss = CrossEntropyMetric(label_smoothing, class_weight.E)
        else:
            self.node_loss = KLDMetric()
            self.edge_loss = KLDMetric()
        self.y_loss = CrossEntropyMetric(label_smoothing, None)
        self.dkl = 0

    def forward(
            self,
            masked_pred_X,
            masked_pred_E,
            pred_y,
            true_X,
            true_E,
            true_y,
            weight,
            q_z_given_x,  # latent_mean,
            p_z  # latent_log_var
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)  未归一化的实数
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean.

        CrossEntropyLoss：输入 logits（内部会自动 softmax）；
        KLDivLoss / 自定义 loss：你自己可能需要先做 softmax，再用对数等操作。"""
        weight_X, weight_E = None, None
        if weight is not None:
            weight_X = weight.unsqueeze(-1)
            weight_E = weight.unsqueeze(-1).unsqueeze(-1)

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(
            masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1)))  # (bs * n * n, de)

        # Remove masked rows 筛选有效位置
        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = (
            self.node_loss(flat_pred_X, flat_true_X, weight=weight_X)
            if true_X.numel() > 0
            else 0.0
        )
        loss_E = (
            self.edge_loss(flat_pred_E, flat_true_E, weight=weight_E)
            if true_E.numel() > 0
            else 0.0
        )
        loss_y = self.y_loss(pred_y, true_y) if pred_y.numel() > 0 else 0.0

        # dkl = -0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
        # loss_mmd = compute_mmd(q_X, p_X, kernel='rbf', sigma=0.5)
        q_z_given_x = q_z_given_x.to_device(device)
        p_z1 = copy.deepcopy(p_z).to_device(device)

        js_x = js_divergence(q_z_given_x.X, p_z1.X)
        js_e = js_divergence(q_z_given_x.E, p_z1.E)
        self.dkl = (js_x + js_e)*0.5

        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y + self.dkl  # 加权

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = (
            self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,  # 节点平均 loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
            "train_epoch/dkl": self.dkl,
        }

        return to_log
