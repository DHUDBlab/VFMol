import os
import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
# import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        # os.makedirs("chains")
    except OSError:
        pass

    # try:
    #     # os.makedirs('checkpoints/' + args.general.name)
    #     # os.makedirs("graphs/" + args.general.name)
    #     os.makedirs("chains/" + args.general.name)
    # except OSError:
    #     pass


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)  # bs,max_node_num,node_type
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    # 给无边的地方标记一个默认的边类型（通常为 type=0），并去除对角线自环（self-loop）
    assert len(E.shape) == 4  # bs,n,n,type
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (  # 去掉对角线上的边
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E  # 把 E[b, i, i, :] 的所有值都设为 0


class PlaceHolder:
    # 用于包装一个批次（batch）的图数据，支持常见的操作比如设备迁移（to_device）、
    # 数据类型转换（type_as）、掩码处理（mask）以及将批次拆分为单个图（split）
    def __init__(self, X, E, y):
        self.X = X  # bs, n, d
        self.E = E  # bs, n, n, e
        self.y = y  # bs,...

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        # self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)  # 类别标签
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1  # # 去掉无效节点，标记-1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:  # 保留张量维度）
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        # 将一个批次的图数据按 node_mask 拆成单独的图。
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list

