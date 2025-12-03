import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torchmetrics import Metric, MeanSquaredError


class CrossEntropyMetric(Metric):
    # 用于累积多个 batch 的交叉熵（Cross Entropy）损失，并求出平均值。
    def __init__(self, label_smoothing, class_weight):
        super().__init__()
        self.label_smoothing = label_smoothing  # 控制标签平滑的程度（缓解过拟合） 0
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.class_weight = class_weight  # 类别权重（用于不平衡数据） None

    def update(self, preds: Tensor, target: Tensor, weight: Tensor = None) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d)."""
        target = torch.argmax(target, dim=-1)
        if weight is not None:
            output = F.cross_entropy(
                preds, target,
                reduction="none",
                label_smoothing=self.label_smoothing,
                weight=None,
            )
            output = (output * weight).sum()
        else:
            output = F.cross_entropy(
                preds, target,
                reduction="sum",
                label_smoothing=self.label_smoothing,
                weight=None,
            )
        # output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


# kld=False KLDMetric 用不上
class KLDMetric(Metric):
    # 用于统计模型输出分布和目标分布之间的 KL 散度（例如 soft target 的训练）
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, weight: Tensor = None) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        模型输出的 log 概率分布（一般需要 log_softmax）？
        target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        # target = torch.argmax(target, dim=-1)
        if weight is not None:
            output = KLDivLoss(reduction="none")(  # 逐元素计算损失
                preds, target,
            )
            output = (output * weight).sum()
        else:
            output = KLDivLoss(reduction="none")(
                preds, target,
            )

        output[output.isnan()] = 0  # zero-out masked places  防止 nan
        output = output.sum()
        # output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class NLL(Metric):
    # 统计所有样本的 负对数似然平均值
    def __init__(self):
        super().__init__()
        self.add_state("total_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()  # 所有元素数量

    def compute(self):
        return self.total_nll / self.total_samples

