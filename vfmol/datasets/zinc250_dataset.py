import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

import vfmol.utils as utils
from vfmol.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from vfmol.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from vfmol.analysis.rdkit_functions import compute_molecular_metrics
from vfmol.datasets.qm9_dataset import (RemoveYTransform, SelectMuTransform, SelectHOMOTransform,
                                        SelectBothTransform, to_list, files_exist)

RDLogger.DisableLog("rdApp.*")


class Zinc250Dataset(InMemoryDataset):
    def __init__(
            self,
            stage,
            root,
            remove_h: bool,
            aromatic: bool,
            target_prop=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        self.target_prop = target_prop
        self.stage = stage
        self.aromatic = aromatic
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)

        self.raw_url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc250k_property.csv'
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        name = 'zinc250k_property.csv'
        return name

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ["proc_tr_no_h.pt", "proc_val_no_h.pt", "proc_test_no_h.pt"]

    def download(self):
        try:
            print('making raw files:', self.raw_dir)
            if not osp.exists(self.raw_dir):
                os.makedirs(self.raw_dir)
            path = download_url(self.raw_url, self.raw_dir)
        except Exception as e:
            print("Cannot download raw files successfully.", e)

    def check_split(self):
        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[0], sep=',', dtype='str')
        # 去掉只有单个原子的行
        dataset = dataset[dataset['smile'].str.len() > 1]  # "C"、"N"、"O" 都是 1个字符长度

        n_samples = len(dataset)
        n_train = 200000  # 249455
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.raw_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.raw_dir, "test.csv"), index=False)

    def process(self):
        self.check_split()

        types = {"C": 0, "N": 1, "O": 2, "F": 3,
                 "P": 4, "S": 5, "Cl": 6, "Br": 7, "I": 8}
        if self.aromatic:
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        else:
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}  # debug

        input_df = pd.read_csv(self.split_paths[self.file_idx], sep=',', dtype='str')
        smile_list = list(input_df["smile"])
        # if isinstance(self.available_prop, list) and self.prop_name in self.available_prop:
        #     prop_list = list(input_df[self.prop_name])

        data_list = []
        for i in tqdm(range(len(smile_list)), desc="Processing SMILES to pyg.Data"):
            smile = smile_list[i]
            # print(smile)
            mol = Chem.MolFromSmiles(smile)
            if not self.aromatic:
                Chem.Kekulize(mol)  # 芳香键明确为单键和双键的交替表示
            if mol is None:
                continue  # 跳过无效的smiles

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]  # 给“无边”保留 0 的编码

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            # 对边按照 (起点, 终点) 的组合排序 起点小的在前面
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            # y = torch.tensor([target_df.loc[i]])
            y = torch.zeros((1, 0), dtype=torch.float)
            # y = mol.GetProp(self.target_prop)

            data = Data(smiles=smile, x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class Zinc250DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir  # data位置
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic

        target = getattr(cfg.general, "target", None)
        regressor = getattr(cfg.general, "conditional", None)  # 条件生成 False
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = SelectBothTransform()
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]  # 获取当前文件的上上级目录，定位项目根目录
        root_path = os.path.join(base_path, self.datadir)  # /defog_codes/data/qm9/
        # DataModule 把多个 Dataset 组织在一起，统一处理 dataloader、batch、shuffle、transform 等。
        datasets = {
            "train": Zinc250Dataset(
                stage="train",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),
            "val": Zinc250Dataset(
                stage="val",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),
            "test": Zinc250Dataset(
                stage="test",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                target_prop=target,
                transform=transform,
            ),  # pyg的Data
        }
        self.test_labels = transform(datasets["test"].data, return_y=True)

        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        super().__init__(cfg, datasets)

        print(super().node_types())
        print(super().node_counts())


class Zinc250infos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic
        self.compute_fcd = cfg.dataset.compute_fcd

        # if cfg.general.conditional:
        #     self.test_labels = datasets["test"].data.y

        self.name = "zinc250"
        if self.remove_h:
            self.atom_encoder = {"C": 0, "N": 1, "O": 2, "F": 3,
                                 "P": 4, "S": 5, "Cl": 6, "Br": 7, "I": 8}
            self.atom_decoder = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
            self.num_atom_types = 9
            self.valencies = [4, 3, 2, 1, 3, 2, 1, 1, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19,
                                 4: 30.97, 5: 30.07, 6: 35.45, 7: 79.9, 8: 126.9}  # 原子质量  # ?
            # 可以用来剔除质量超限的分子，或者在训练中对 mass 进行标准化
            self.max_n_nodes = 38
            self.max_weight = 350
            self.n_nodes = torch.tensor(
                [0, 0, 0, 0, 0, 0,
                 1.3362e-05, 2.2271e-05, 7.1266e-05, 3.0734e-04, 7.5275e-04, 2.9486e-03,
                 4.7481e-03, 7.1667e-03, 1.1394e-02, 1.7269e-02, 2.5322e-02, 3.4711e-02,
                 4.6720e-02, 5.8216e-02, 7.0950e-02, 8.2152e-02, 7.4571e-02, 8.4304e-02,
                 9.2677e-02, 9.1519e-02, 7.7386e-02, 6.3151e-02, 4.0199e-02, 3.1241e-02,
                 2.4511e-02, 1.9090e-02, 1.4930e-02, 1.0356e-02, 7.0776e-03, 4.0889e-03,
                 1.5812e-03, 5.4786e-04, 4.4541e-06]
            )
            self.node_types = torch.tensor([0.73675, 0.12212, 9.9743e-02, 1.3784e-02, 2.2250e-05, 1.7806e-02,
                                            7.4098e-03, 2.2043e-03, 1.5575e-04])
            if self.aromatic:
                self.edge_types = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])
            else:
                self.edge_types = torch.tensor([0.90657, 6.9431e-02, 2.3763e-02, 2.3417e-04])  # debug

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:7] = torch.tensor(
                [0.0, 0.11365, 0.30450, 0.35050, 0.22652, 2.0738e-05, 4.8175e-03]
            )


def get_smiles(cfg, datamodule, dataset_infos, evaluate_datasets=False):
    return {
        "train": get_loader_smiles(
            cfg,
            datamodule.train_dataloader(),
            dataset_infos,
            "train",
            evaluate_dataset=evaluate_datasets,
        ),
        "val": get_loader_smiles(
            cfg,
            datamodule.val_dataloader(),
            dataset_infos,
            "val",
            evaluate_dataset=evaluate_datasets,
        ),
        "test": get_loader_smiles(
            cfg,
            datamodule.test_dataloader(),
            dataset_infos,
            "test",
            evaluate_dataset=evaluate_datasets,
        ),
    }


def get_loader_smiles(
        cfg,
        dataloader,
        dataset_infos,
        split_key,
        evaluate_dataset=False,
):
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = f"{split_key}_smiles_no_h.npy"

    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print(f"Dataset {split_key} smiles were found.")
        smiles = np.load(smiles_path).tolist()
    else:
        print(f"Computing dataset {split_key} smiles...")
        smiles = compute_zinc250_smiles(atom_decoder, dataloader, remove_h)
        np.save(smiles_path, np.array(smiles))  # npy存smiles

    return smiles


def compute_zinc250_smiles(atom_decoder, train_dataloader, remove_h):
    """
    :param dataset_name: qm9 or qm9_second_half
    :return:
    把 训练数据 中每一个图（分子）转换成 SMILES 表示，并统计无效/非连通分子数量。
    """
    print(f"\tConverting Zinc250k dataset to SMILES ...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]  # 计算每个图（分子）的节点数

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(  # 考虑部分电荷
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print(
                "\tConverting Zinc250k dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
