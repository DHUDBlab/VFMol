import torch

from vfmol import utils


class NoiseDistribution:
    def __init__(self, model_transition, dataset_infos):
        # 定义图上节点、边的离散噪声分布，以及它们的极限分布（limit distribution）
        self.x_num_classes = dataset_infos.output_dims["X"]
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        # 初始化新增类别数为 0
        # self.x_added_classes = 0
        # self.e_added_classes = 0
        # self.y_added_classes = 0
        self.transition = model_transition  # 'marginal'

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "argmax":
            node_types = dataset_infos.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "marginal":

            node_types = dataset_infos.node_types.float()  # torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
            x_limit = node_types / torch.sum(node_types)  # 节点类型的概率分布

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # TODO: dummy?
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

    def get_limit_dist(self):
        # 获取之前设定的节点/边/图标签的极限分布 边际分布
        return self.limit_dist

    def get_noise_dims(self):
        return {  # 返回当前节点/边/图标签的类别数（包括虚拟类别）
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),  # # 注意：这里 `y` 用的是 `E` 的长度，可能是临时代码
        }
