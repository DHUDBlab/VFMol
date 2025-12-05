import torch
from vfmol import utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(  # 基于价电子期望和边连接数推算“过多/过少电子”。
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies
        )
        self.valency = ValencyFeature()  # 从 bond strength（边）中计算每个原子的当前电子数量。
        self.weight = WeightFeature(  # 整个分子的总质量
            max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights
        )

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)  # (bs, n, 1) 每个节点的“电子差异”特征
        valency = self.valency(noisy_data).unsqueeze(-1)  # (bs, n, 1) 每个节点的价电子数特征
        weight = self.weight(noisy_data)  # (bs, 1) 整个图（分子）的归一化质量

        # 目前没有添加任何新的边特征，所以返回形状为 (bs, n, n, 0) 的空 tensor
        extra_edge_attr = torch.zeros((*noisy_data["E_t"].shape[:-1], 0)).type_as(noisy_data["E_t"])

        return utils.PlaceHolder(
            X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight
        )


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        # 从图数据中计算“电荷相关的特征”
        # 估算“价态偏差”（即：当前成键情况和理想价态之间的差异）
        self.remove_h = remove_h
        self.valencies = valencies  # [4, 3, 2, 1]

    def __call__(self, noisy_data):  # example_data xt，et，yt

        de = noisy_data["E_t"].shape[-1]
        if de == 5:
            bond_orders = torch.tensor(
                [0, 1, 2, 3, 1.5], device=noisy_data["E_t"].device
            ).reshape(1, 1, 1, -1)
        else:
            bond_orders = torch.tensor(
                [0, 1, 2, 3], device=noisy_data["E_t"].device
            ).reshape(1, 1, 1, -1)
        weighted_E = noisy_data["E_t"] * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)
        # 不合理
        # bond_strength = (noisy_data["E_t"] * bond_orders).sum(dim=-1)  # (bs, n, n)
        # current_valencies = bond_strength.sum(dim=-1)  # (bs, n)

        valencies = torch.tensor(
            self.valencies, device=noisy_data["X_t"].device
        ).reshape(1, 1, -1)
        X = noisy_data["X_t"] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)  # (bs, n)
        # 或许也该是.sum()?

        return (normal_valencies - current_valencies).type_as(noisy_data["X_t"])


class ValencyFeature:
    # 从 E_t 中估算每个节点的价态
    def __call__(self, noisy_data):
        de = noisy_data["E_t"].shape[-1]
        if de == 5:
            bond_orders = torch.tensor(
                [0, 1, 2, 3, 1.5], device=noisy_data["E_t"].device
            ).reshape(1, 1, 1, -1)
        else:
            bond_orders = torch.tensor(
                [0, 1, 2, 3], device=noisy_data["E_t"].device
            ).reshape(1, 1, 1, -1)
        # bond_orders = torch.tensor([0, 1, 2, 3], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)  # debug
        E = noisy_data["E_t"] * bond_orders  # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)  # (bs, n)
        # valencies = E.sum(dim=-1).sum(dim=-1)  # ✅ 先对边类型求加权和，再对邻居求和
        return valencies.type_as(noisy_data["X_t"])


class WeightFeature:
    # 计算分子的总原子质量，并将它归一化（除以最大质量）作为一个特征输出
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight  # 150
        # 把字典的值取出来变成 tensor，比如 [12, 1, 16]， 其中每一维对应 原子类型的质量
        self.atom_weight_list = torch.tensor(list(atom_weights.values()))

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data["X_t"], dim=-1)  # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]  # (bs, n)
        return (
            X_weights.sum(dim=-1).unsqueeze(-1).type_as(noisy_data["X_t"])  # 得到总质量 (bs,)->(bs, 1)
            / self.max_weight
        )  # (bs, 1)
