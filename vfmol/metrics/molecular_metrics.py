from torchmetrics import MeanSquaredError, MeanAbsoluteError
import time
import warnings
import fcd
# packages for visualization
import torch
from torchmetrics import Metric
# import wandb
import torch.nn as nn

from vfmol.analysis.rdkit_functions import compute_molecular_metrics


class SamplingMolecularMetrics(nn.Module):
    # 在模型训练或测试过程中评估生成分子的结构特性是否和真实分子数据一致（比如：原子数量分布、键类型分布、原子价等）
    def __init__(self, dataset_infos, dataset_smiles, cfg):
        super().__init__()
        di = dataset_infos  # 是用定义的分布算的
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)

        num_node_types = di.output_dims["X"]
        self.generated_node_dist = GeneratedNodesDistribution(num_node_types)

        num_edge_types = di.output_dims["E"]
        self.generated_edge_dist = GeneratedEdgesDistribution(num_edge_types)
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)
        self.cfg = cfg

        # 设置目标分布（用于对比）
        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer("n_target_dist", n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer("node_target_dist", node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer("edge_target_dist", edge_target_dist)

        valency_target_dist = di.valency_distribution.type_as(
            self.generated_valency_dist.edgepernode_dist
        )
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer("valency_target_dist", valency_target_dist)
        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = dataset_smiles["train"]
        self.val_smiles = dataset_smiles["val"]
        self.test_smiles = dataset_smiles["test"]
        self.dataset_info = di

    def forward(
            self,
            molecules: list,
            ref_metrics,
            name,
            current_epoch,
            val_counter,
            local_rank,
            test=False,
            labels=None,
    ):
        stability, rdkit_metrics, all_smiles, to_log = compute_molecular_metrics(
            molecules, self.train_smiles, self.dataset_info, labels, self.cfg, test
        )

        self.dataset_info.compute_fcd = True
        if self.dataset_info.compute_fcd:
            to_log["fcd"] = compute_fcd(
                val_smiles=self.test_smiles if test else self.val_smiles,
                generated_smiles=all_smiles,
            )
        else:
            print("FCD computation is disabled. Skipping.")
            to_log["fcd"] = -1

        print('fcd', to_log['fcd'])

        print("Starting custom metrics")  # 定制评估指标模块（custom metrics）
        # 主要作用是对比生成分子和数据集中真实分子的结构统计特征，判断模型是否“生成得像”
        self.generated_n_dist.update(molecules)  # 生成分子的原子数量分布
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist.update(molecules)  # 各类型原子的比例（如 C, N, O）
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist.update(molecules)  # 根据生成分子统计分布
        generated_edge_dist = self.generated_edge_dist.compute()  # 计算频率（归一化分布）
        self.edge_dist_mae(generated_edge_dist)  # 与真实分布计算差距。

        self.generated_valency_dist.update(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

        for i, atom_type in enumerate(self.dataset_info.atom_encoder.keys()):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                    generated_probability - target_probability
            ).item()

        for j, bond_type in enumerate(
                ["No bond", "Single", "Double", "Triple", "Aromatic"]
                # ["No bond", "Single", "Double", "Triple"]
        ):
            if j < len(generated_edge_dist):
                generated_probability = generated_edge_dist[j]
                target_probability = self.edge_target_dist[j]
                to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                        generated_probability - target_probability
                ).item()

        for valency in range(6):  # 用连接的边计算的化学价
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                    generated_probability - target_probability
            ).item()

        if local_rank == 0:
            print("Custom metrics computed.")
            print("Stability metrics:", stability, "--", rdkit_metrics[0])

        return to_log

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()


def compute_fcd(val_smiles, generated_smiles):
    """smiles have must be a list of str"""

    print("Starting FCD computation")
    start = time.time()

    # not using fcd.canonical_smiles because both smiles are already in canonical form (result from the
    # Chem.MolToSmiles)
    # filter out None values (not sanitizable molecules)
    generated_smiles = [smile for smile in generated_smiles if smile is not None]

    # supress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            fcd_score = fcd.get_fcd(generated_smiles, val_smiles)
        except Exception as e:
            print(f"Error in FCD computation. Setting FCD to -1.")
            fcd_score = -1

    end = time.time()
    print("FCD computation time:", end - start, "FCD score is", fcd_score)

    return fcd_score


class GeneratedNDistribution(Metric):
    # 节点个数分布（每个分子有多少个原子）
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),  # 0 到 max_n 个原子
            dist_reduce_fx="sum",  # 在分布式训练中，会自动把所有进程的统计加在一起
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    # 节点类型分布（统计不同原子的出现频率）
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                        int(atom_type) != -1  # 忽略了 -1（表示 padding 或 masked 原子）
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    # 边（化学键）类型的分布
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()  # 排除重复边，自连接边
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)  # 统计每种边类型的数量
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class ValencyDistribution(Metric):
    # 评估分子中各原子的“价”（valency）分布 的指标，也就是每个原子连接边的类型总和
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            edge_types[edge_types == 5] = 0.0  # zero out virtual states
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count
                # 注意：valency 是浮点数，在 PyTorch 中是非法的——你原始代码中这一行会报错
                # self.edgepernode_dist[int(valency)] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    #  直方图之间的平均绝对误差（MAE）指标，用于衡量 一个预测分布与目标分布（target_histogram）之间的距离
    #  输出的 MAE 描述了这两个分布在每个 bin（类别）上的差异平均值
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3  # 必须归一化为总和为 1
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)

