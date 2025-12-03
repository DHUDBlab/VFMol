import copy
import os

import torch
from torch_geometric.data.lightning import LightningDataset
from tqdm import tqdm

import vfmol.utils as utils
from vfmol.datasets.dataset_utils import load_pickle, save_pickle


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        建模并采样“节点数目”的分布（比如分子图中节点个数），常用于图生成任务中，先预测/采样 要生成多少个节点
        historgram: dict. The keys are num_nodes, the values are counts  一个字典：{num_nodes: count}
        """
        if type(histogram) == dict:
            # 将字典转换为一个离散概率向量 prob，索引表示节点个数，值表示该节点数的出现次数
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()  # 归一化为概率分布
        self.m = torch.distributions.Categorical(prob)  # 定义一个多项式采样器

    def sample_n(self, n_samples, device):
        # 从这个节点数分布中采样 n 个图的节点数（即：生成多少个节点）
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        # 计算给定的节点数列表在该分布下的对数概率
        assert len(batch_n_nodes.size()) == 1  # 确保是一维向量 eg: torch.tensor([5, 6, 5, 7])
        p = self.prob.to(batch_n_nodes.device)  # 把概率分布挪到相同设备

        probas = p[batch_n_nodes]  # 取出每个样本对应的概率值
        log_p = torch.log(probas + 1e-30)
        return log_p


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size if "debug" not in cfg.general.name else 2,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None
        print(f'This dataset contains {len(datasets["train"])} training graphs, {len(datasets["val"])} validation '
              f'graphs, {len(datasets["test"])} test graphs.')

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=1000):
        # 统计所有分子图（batch 中的图）中，每个图的节点数分布
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)  # 统计每张图有多少个节点
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())  # 只保留出现过的节点数段（去掉末尾全 0 的）
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        # 统计每种 节点类型（原子类型） 的频率分布
        num_classes = None

        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    # 统计训练集中所有原子出现的“原子价”（valency）分布
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected
        # 每两个节点最多有一个三键（triple bond），即每个原子最多连接其他节点的数量是 max_n_nodes - 1

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3])  # 键类型对应的“价值” [0, 1, 2, 3, 1.5]

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]  # 取出该原子发出的所有边的属性
                edges_total = edges.sum(dim=0)  # 各种键类型的总数 edges_total = [2, 1, 0, 0] 单键2次，双键1次
                valency = (edges_total * multiplier).sum()  # valency = 2*1 + 1*2 = 4
                valencies[valency.long().item()] += 1  # valencies[4] += 1
        valencies = valencies / valencies.sum()  # 归一化为概率分布
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        # 自动计算模型的输入输出维度
        # datamodule: 包含 train_dataloader()，用于拿到一个 batch 的样本
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(  # 把稀疏图转为 dense 格式（带 X、E 和 node_mask）
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )  # PlaceHolder(X=X, E=E, y=None), node_mask

        example_data = {  # 构造一个样本 example_data 字典（方便后续特征抽取）
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": example_batch["x"].size(1),  # 节点特征维度
            "E": example_batch["edge_attr"].size(1),  # 边特征维度
            "y": example_batch["y"].size(1) + 1,  # this part take into account the conditioning
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }

    def compute_reference_metrics(self, datamodule, sampling_metrics):
        # 计算并保存用于评估图生成质量的“参考指标（reference metrics）
        # datamodule：PyTorch Lightning 风格的数据模块，包含 train_dataloader() 等方法。
        # sampling_metrics：用于评估生成图结构质量的评估器

        ref_metrics_path = os.path.join(  # 保存参考指标的路径
            datamodule.train_dataloader().dataset.root, f"ref_metrics_no_h.pkl"
        )

        # Only compute the reference metrics if they haven't been computed already
        if not os.path.exists(ref_metrics_path):

            print("Reference metrics not found. Computing them now.")
            # Transform the training dataset into a list of graphs in the appropriate format
            training_graphs = []
            print("Converting training dataset to format required by sampling metrics.")
            for data_batch in tqdm(datamodule.train_dataloader()):
                dense_data, node_mask = utils.to_dense(
                    data_batch.x,
                    data_batch.edge_index,
                    data_batch.edge_attr,
                    data_batch.batch,
                )
                dense_data = dense_data.mask(node_mask, collapse=True).split(node_mask)
                for graph in dense_data:
                    training_graphs.append([graph.X, graph.E])

            # defining dummy arguments 设置假参数传入指标模块
            dummy_kwargs = {
                "name": "ref_metrics",
                "current_epoch": 0,
                "val_counter": 0,
                "local_rank": 0,
                "ref_metrics": {"val": None, "test": None},
            }

            print("Computing validation reference metrics.")
            # do not have to worry about wandb because it was not init yet
            val_sampling_metrics = copy.deepcopy(sampling_metrics)

            val_ref_metrics = val_sampling_metrics.forward(
                training_graphs,
                test=False,
                **dummy_kwargs,
            )

            print("Computing test reference metrics.")
            test_sampling_metrics = copy.deepcopy(sampling_metrics)
            test_ref_metrics = test_sampling_metrics.forward(
                training_graphs,
                test=True,
                **dummy_kwargs,
            )

            print("Saving reference metrics.")
            # print(f"deg: {test_reference_metrics['degree']} | clus: {test_reference_metrics['clustering']} | orbit:
            # {test_reference_metrics['orbit']}") breakpoint()
            save_pickle(
                {"val": val_ref_metrics, "test": test_ref_metrics}, ref_metrics_path
            )

        print("Loading reference metrics.")
        self.ref_metrics = load_pickle(ref_metrics_path)
        print("Validation reference metrics:", self.ref_metrics["val"])
        print("Test reference metrics:", self.ref_metrics["test"])
