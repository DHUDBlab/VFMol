import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import numpy as np
import rdkit.Chem
# import wandb
import matplotlib.pyplot as plt


class MolecularVisualization:
    def __init__(self, dataset_infos):
        # 将图结构的分子表示（节点和邻接矩阵）转化为 RDKit 分子结构，并将其 可视化为图像（PNG 或 GIF）
        self.dataset_infos = dataset_infos  # 提供原子类型的字典

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        将图（节点列表 + 邻接矩阵）转为 RDKit 分子
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()  # 将图（节点列表 + 邻接矩阵）转为 RDKit 分子
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(
        self, path: str, molecules: list, num_molecules_to_visualize: int, log="graph"
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)

        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)

            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # 将多个分子图结构转为 RDKit 分子对象。
        # 用最后一个分子来生成一个 2D 坐标布局（Compute2DCoords）。
        # 将所有分子对齐到这个布局上（保持结构一致）。
        # 每帧保存一个 PNG 图像，最后合成为 GIF 动画。
        # 也尝试画一个静态的“grid”图（所有分子排成一个表格）
        RDLogger.DisableLog("rdApp.*")
        # convert graphs to the rdkit molecules
        mols = [
            self.mol_from_graphs(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            Draw.MolToFile(
                mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}"
            )
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(
                os.path.join(path, "{}_grid_image.png".format(path.split("/")[-1]))
            )
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols
