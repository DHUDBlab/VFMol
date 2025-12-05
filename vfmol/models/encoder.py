import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch import Tensor
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn import (GINEConv, BatchNorm)

from vfmol.utils import PlaceHolder

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GINEMLP(nn.Module):
    def __init__(
            self, hidden: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.in_channels = hidden  # used by GINEConv
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            BatchNorm(2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            BatchNorm(hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GraphEncoder(nn.Module):
    def __init__(self, dataset_max_n, num_node_types, num_edge_types, latent_dim):
        super(GraphEncoder, self).__init__()
        self.dataset_max_n = dataset_max_n
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.latent_dim = latent_dim
        self.hidden_encoder = 32
        self.num_layers = 3

        self.graph_layers = nn.ModuleList([
            GINEConv(  # Graph Isomorphism Network with Edge features Convolution
                GINEMLP(self.hidden_encoder),  # GIN Enhanced MLP，用于节点特征的非线性转换
                train_eps=True,  # 是否训练GIN的epsilon参数
                edge_dim=self.hidden_encoder,  # 边特征的维度
            )
            for _ in range(self.num_layers)
        ])

        self.edge_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_encoder, self.hidden_encoder),
                nn.ReLU(),
                nn.Linear(self.hidden_encoder, self.hidden_encoder)
            )
            for _ in range(self.num_layers - 1)
        ])

        self.node_embed = nn.Sequential(
            nn.Linear(self.num_node_types, self.hidden_encoder),
            BatchNorm(self.hidden_encoder),  # 对批次归一化
            nn.ReLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(self.num_edge_types, self.hidden_encoder),
            BatchNorm(self.hidden_encoder),  # 对批次归一化
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(3 * 2 * self.hidden_encoder, 2 * self.latent_dim)  # self.latent_dim
        # self.dropout = nn.Dropout(0.05)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        out = []
        x = self.node_embed(x)
        edge_attr_real = self.edge_embed(edge_attr)

        for i in range(self.num_layers):
            new_x = x  # self.norm_layers[i](x)
            new_x = self.graph_layers[i](new_x, edge_index, edge_attr_real)  # N_tot x hidden
            x = new_x + x  # res 要吗

            new_out = torch.cat([tgnn.global_mean_pool(new_x, batch),  # bs,hidden
                                 tgnn.global_max_pool(new_x, batch)], dim=1)
            out.append(new_out)

            if i != self.num_layers - 1:
                src, dst = edge_index  # 源节点和目标节点索引
                edge_input = torch.cat([x[src], x[dst], edge_attr_real], dim=-1)  #
                edge_attr_real = nn.ReLU()(self.edge_layers[i](edge_input)) + edge_attr_real

        out = torch.cat(out, dim=1)
        # print("out:", out.size())
        out = self.fc(out)

        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]

        return mu, log_var


class GraphEncoder2(nn.Module):
    def __init__(self, dataset_max_n, num_node_types, num_edge_types, latent_dim):
        super(GraphEncoder2, self).__init__()
        self.dataset_max_n = dataset_max_n
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.latent_dim = latent_dim
        self.hidden_encoder = 32
        self.num_layers = 3

        self.graph_layers = nn.ModuleList([
            GINEConv(  # Graph Isomorphism Network with Edge features Convolution
                GINEMLP(self.hidden_encoder),  # GIN Enhanced MLP，用于节点特征的非线性转换
                train_eps=True,  # 是否训练GIN的epsilon参数
                edge_dim=self.hidden_encoder,  # 边特征的维度
            )
            for _ in range(self.num_layers)
        ])

        self.edge_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_encoder, self.hidden_encoder),
                nn.ReLU(),
                nn.Linear(self.hidden_encoder, self.hidden_encoder)
            )
            for _ in range(self.num_layers - 1)
        ])

        self.node_embed = nn.Sequential(
            nn.Linear(self.num_node_types, self.hidden_encoder),
            BatchNorm(self.hidden_encoder),  # 对批次归一化
            nn.ReLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(self.num_edge_types, self.hidden_encoder),
            BatchNorm(self.hidden_encoder),  # 对批次归一化
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 2 * self.hidden_encoder, self.latent_dim),
            nn.LeakyReLU(),
        )
        self.mlp_X = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_node_types),
            nn.Softmax(-1),
        )
        self.mlp_E = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_edge_types),
            nn.Softmax(-1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        out = []
        x = self.node_embed(x)
        edge_attr_real = self.edge_embed(edge_attr)

        for i in range(self.num_layers):
            new_x = x  # self.norm_layers[i](x)
            new_x = self.graph_layers[i](new_x, edge_index, edge_attr_real)  # N_tot x hidden
            x = new_x + x  # res 要吗

            new_out = torch.cat([tgnn.global_mean_pool(new_x, batch),  # bs,hidden
                                 tgnn.global_max_pool(new_x, batch)], dim=1)
            out.append(new_out)

            if i != self.num_layers - 1:
                src, dst = edge_index  # 源节点和目标节点索引
                edge_input = torch.cat([x[src], x[dst], edge_attr_real], dim=-1)  #
                edge_attr_real = nn.ReLU()(self.edge_layers[i](edge_input)) + edge_attr_real

        out = torch.cat(out, dim=1)
        # print("out:", out.size())
        z = self.fc(out)
        qz_X = self.mlp_X(z)
        qz_E = self.mlp_E(z)
        qz_given_x = PlaceHolder(X=qz_X, E=qz_E, y=None)

        return qz_given_x


class MLPGenerator(nn.Module):
    def __init__(self, dataset_max_n, latent_dim, num_node_types):
        super().__init__()
        self.dataset_max_n = dataset_max_n
        self.latent_channels = latent_dim
        self.spatial_dim = 64
        self.node_type = num_node_types

        self.set_channels = 64
        self.hidden_set_gen = 32
        self.mlp_gen_hidden = 512
        self.hidden_decoder = 64
        self.hidden, hidden_final = 64, 128
        self.dim_feedforward_transformer = 2 * self.hidden
        self.set_channels = 64
        self.decoder_layer_num = 2  # 对应["Transformer", "Transformer"]
        self.n_heads_transformer = 8

        self.softmax = nn.Softmax(dim=-1)  # softmax 的对数概率张量

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_channels, self.dataset_max_n * self.set_channels),
            nn.GELU()
        )
        self.first_set_layer = nn.Linear(self.set_channels, self.hidden_decoder)
        self.last_set_layer = torch.nn.Linear(self.hidden_decoder, self.spatial_dim)  # hidden_final
        self.atom_type_layer = nn.Sequential(nn.Linear(self.spatial_dim, hidden_final),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_final, self.node_type),
                                             )

        self.decoder_layers = nn.ModuleList()
        for i in range(self.decoder_layer_num):  # ["Transformer", "Transformer"]
            module = TransformerEncoderLayer(self.hidden, self.n_heads_transformer,
                                             self.dim_feedforward_transformer, batch_first=True)
            # hidden 输入和输出的特征维度  dim_feedforward 前馈神经网络的隐藏层维度。通常是 d_model 的 2～4 倍
            self.decoder_layers.append(module)

    def forward(self, latent: Tensor, node_num_mask):
        batch_size = latent.shape[0]

        points = self.mlp(latent).reshape(batch_size, self.dataset_max_n, self.set_channels)

        # x = x * node_num_mask.unsqueeze(-1)  # 无效节点mask  bs,max_node,1
        x = self.first_set_layer(points)

        for i in range(self.decoder_layer_num):
            x = self.decoder_layers[i](x, src_key_padding_mask=~node_num_mask)

        x = x * node_num_mask.unsqueeze(-1)
        positions = self.last_set_layer(x)  # bs x n x spatial_dim 节点位置
        log_atom_types = self.softmax(self.atom_type_layer(positions))  # 节点类型

        return positions, log_atom_types


class GraphDecoder(nn.Module):
    def __init__(self, dataset_max_n, num_node_types, num_edge_types):
        super().__init__()
        self.dataset_max_n = dataset_max_n
        self.node_type = num_node_types
        self.edge_type = num_edge_types
        self.hidden_decoder = 64
        self.hidden, hidden_final = 64, 128
        self.dim_feedforward_transformer = 2 * self.hidden
        self.set_channels = 64
        self.spatial_dim = 64
        self.softmax = nn.Softmax(dim=-1)  # softmax 的对数概率张量

        self.mlp_type = nn.Sequential(
            nn.Linear(2 * (self.node_type + self.hidden_decoder), self.node_type + self.hidden_decoder),
            nn.LeakyReLU(),
            nn.Linear(self.node_type + self.hidden_decoder, self.edge_type),
        )

    def forward(self, positions, log_atom_types, node_num_mask):
        x_convert = torch.cat((positions, log_atom_types), dim=-1)  # node_feature+ atom_types
        # print("x_convert", x_convert.shape)  # 输出形状为 (batch, node_num, spatial_dim)
        x_convert = x_convert * node_num_mask.unsqueeze(-1)  # 只保留真实节点的特征

        positions_4d = x_convert.unsqueeze(2).expand(-1, -1, self.dataset_max_n, -1)
        edge_prob_input = torch.cat((positions_4d, positions_4d.transpose(1, 2)), dim=3)
        # print("edge_prob_input", edge_prob_input.size())

        edge_type = self.mlp_type(edge_prob_input)
        log_edge_type = self.softmax(edge_type)  # 边类型
        # print("log_edge_type", log_edge_type.size())
        edge_pred = (log_edge_type + log_edge_type.transpose(1, 2)) / 2  # i,j = j,i
        # print("edge_pred", edge_pred.size())

        out = PlaceHolder(X=log_atom_types, E=edge_pred, y=None).mask(node_num_mask)
        return out


class GraphTransformerVae(nn.Module):
    def __init__(self, dataset_max_n, num_node_types, num_edge_types, latent_dim):
        super().__init__()
        self.node_type = num_edge_types
        self.num_edge_types = num_node_types
        self.latent_dim = latent_dim
        self.dataset_max_n = dataset_max_n

        self.encoder = GraphEncoder(self.dataset_max_n, self.node_type, self.num_edge_types, self.latent_dim)
        self.set_generator = MLPGenerator(self.dataset_max_n, self.latent_dim, self.num_edge_types)
        self.decoder = GraphDecoder(self.dataset_max_n, self.node_type, self.num_edge_types)
        self.normal = torch.distributions.normal.Normal(0.0, 1.0)

    def forward(self, data, node_mask):
        latent_mean, log_var = self.encoder(data)

        latent = self.reparameterize(latent_mean, log_var)
        positions, log_atom_types = self.set_generator(latent, node_mask)

        out = self.decoder(positions, log_atom_types, node_mask)
        # log_atom_types, edge_pred
        return [out, latent_mean, log_var]

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std
        return z


if __name__ == "__main__":
    # 先验分布的KL散度
    # dkl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # losses.append(bert * dkl / avg_n)
    pass
