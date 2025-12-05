import time
import os
import copy

import numpy as np
import pickle
# import matplotlib.pyplot as plt
from rdkit import RDLogger
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical

from models.transformer_model import GraphTransformer
from models.encoder import GraphEncoder2
from vfmol.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete

from vfmol import utils
from vfmol.flow_matching.noise_distribution import NoiseDistribution
from vfmol.flow_matching.time_distorter import TimeDistorter

RDLogger.DisableLog("rdApp.*")


class DiscreteFlowMatching(pl.LightningModule):
    def __init__(
            self,
            cfg,
            dataset_infos,
            train_metrics,
            sampling_metrics,
            visualization_tools,
            extra_features,
            domain_features,
            test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"

        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps  # 500
        self.sample_T = (  # test 1000，train 500
            cfg.sample.sample_steps if cfg.general.test_only is not None else self.T
        )
        self.eta = self.cfg.sample.eta  # 0
        self.conditional = cfg.general.conditional  # False
        self.test_labels = test_labels  # None
        self.latent_dim = 64

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        print('max num nodes', len(self.node_dist.prob) - 1)  # 0-9,不要0
        print('min num nodes', torch.where(self.node_dist.prob > 0)[0][0])

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)  # 'marginal'
        self.limit_dist = self.noise_dist.get_limit_dist()  # 噪声的 初始 x0，e0

        if self.cfg.model.weighted_loss:  # False
            class_weight = utils.PlaceHolder(
                X=1 / (self.limit_dist.X + 1e-4),
                E=1 / (self.limit_dist.E + 1e-4),
                y=None,
            )
            class_weight.X = torch.sqrt(class_weight.X)
            class_weight.E = torch.sqrt(class_weight.E)
        else:
            class_weight = utils.PlaceHolder(X=None, E=None, y=None)

        self.train_loss = TrainLossDiscrete(  # node和edge的交叉熵
            self.cfg.model.lambda_train,  # [5, 0]
            self.cfg.model.label_smoothing,  # 0.0
            class_weight=class_weight,  # None
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.encoder = GraphEncoder2(
            dataset_max_n=self.dataset_info.max_n_nodes,
            num_node_types=self.dataset_info.num_atom_types,
            num_edge_types=len(self.dataset_info.edge_types),  # 带无边
            latent_dim=self.latent_dim
        )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.adapt_counter = 0

        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            s=cfg.train.mode_s,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        if self.conditional:
            # 执行 条件生成 时的一个数据扰动或数据增强策略
            if torch.rand(1) < 0.1:  # 以10%的概率，执行下面的“干扰”操作
                # condition = torch.ones_like(data.y, device=self.device) * -1
                data.y = torch.ones_like(data.y, device=self.device) * -5  # 把 data.y（本来的条件标签）全部替换为 -1

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        qz_given_x = self.encoder(data)

        noisy_data = self.apply_noise(X, E, data.y, node_mask, qz_given_x)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)  # 预测X1，E1,y1

        # weight = noisy_data["t"]
        weight = None

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            weight=weight,
            q_z_given_x=qz_given_x,
            p_z=self.limit_dist
        )  # 反向传播（backprop）和优化器更新参数

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        # 在 训练流程正式开始前 运行
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")

        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" dkl: {100 * to_log['train_epoch/dkl'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )

    def on_validation_epoch_start(self) -> None:
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        qz_given_x = self.encoder(data)

        noisy_data = self.apply_noise(X, E, data.y, node_mask, qz_given_x)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        weight = None

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=dense_data.X,
            true_E=dense_data.E,
            true_y=data.y,
            weight=weight,
            q_z_given_x=qz_given_x,
            p_z=self.limit_dist
        )
        loss = 100 * loss
        self.log("val_epoch_NLL", loss, sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        print("finishing the validation step for the experiment: ")

        val_nll = self.trainer.callback_metrics.get("val_epoch_NLL")
        self.print(
            f"Epoch {self.current_epoch}: Val Loss {val_nll :.2f}"
        )

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            print("Starting to sample")
            to_log = self.sample_and_evaluate(test=False)

            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Done validating.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""

        results_df = pd.DataFrame()
        # search = False
        if self.cfg.general.search == "search":  # False
            num_step_list = [1, 5, 10, 50, 100, 1000]
            if self.cfg.dataset.name == "qm9":
                num_step_list = [1, 5, 10, 50, 100, 500]
            if self.cfg.dataset.name == "guacamol":
                num_step_list = [1, 5, 10, 50, 100, 500]
            if self.cfg.dataset.name == "moses":
                num_step_list = [1, 5, 10, 50, 100, 500]
            results_df = pd.DataFrame()
            for num_step in num_step_list:
                for distortor in ["identity", "cos", "revcos", "polyinc", "polydec"]:
                    self.cfg.sample.time_distortion = distortor
                    print(
                        f"############# Testing distortor: {distortor}, num steps: {num_step} #############"
                    )
                    self.sample_T = num_step
                    res = self.sample_and_evaluate(
                        test=True, rdb="general", rdb_crit="general"
                    )
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["distortor"] = distortor
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(f"test_epoch{self.current_epoch}_distortor.csv")

            results_df.reset_index(inplace=True)
            results_df.set_index(["num_step", "distortor"], inplace=True)
            results_df.to_csv(f"test_epoch{self.current_epoch}_distortor.csv")

            self.cfg.sample.time_distortion = 'identity'
            results_df = pd.DataFrame()
            for num_step in num_step_list:
                for omega in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    for distortor in ["identity"]:
                        # for distortor in ["polydec"]:
                        self.cfg.sample.time_distortion = distortor
                        self.cfg.sample.omega = omega
                        print(
                            f"############# Testing omega: {omega}, num steps: {num_step}, distortor: {distortor} #############"
                        )
                        self.sample_T = num_step
                        res = self.sample_and_evaluate(
                            test=True, rdb="general", rdb_crit="general"
                        )
                        mean_res = {f"{key}_mean": res[key][0] for key in res}
                        std_res = {f"{key}_std": res[key][1] for key in res}
                        mean_res.update(std_res)
                        res_df = pd.DataFrame([mean_res])
                        res_df["num_step"] = num_step
                        res_df["omega"] = omega
                        res_df["distortor"] = distortor
                        results_df = pd.concat([results_df, res_df], ignore_index=True)
                        results_df.to_csv(f"test_epoch{self.current_epoch}_omega.csv")

            results_df.reset_index(inplace=True)
            results_df.set_index(["num_step", "omega", "distortor"], inplace=True)
            results_df.to_csv(f"test_epoch{self.current_epoch}_omega.csv")

            self.cfg.sample.omega = 0.0
            results_df = pd.DataFrame()
            for num_step in num_step_list:
                for eta in [0.0, 5, 10, 25, 50, 100, 200]:
                    for distortor in ["identity"]:
                        # for distortor in ["polydec"]:
                        print(
                            f"############# Testing eta: {eta}, num steps: {num_step} #############"
                        )
                        self.sample_T = num_step
                        self.cfg.sample.eta = eta
                        self.cfg.sample.time_distortion = distortor
                        res = self.sample_and_evaluate(
                            # test=True, rdb="column", rdb_crit="x_1", eta=eta
                            test=True,
                            rdb="general",
                            rdb_crit="general",
                            eta=eta,
                        )
                        mean_res = {f"{key}_mean": res[key][0] for key in res}
                        std_res = {f"{key}_std": res[key][1] for key in res}
                        mean_res.update(std_res)
                        res_df = pd.DataFrame([mean_res])
                        res_df["num_step"] = num_step
                        res_df["eta"] = eta
                        res_df["distortor"] = distortor
                        results_df = pd.concat([results_df, res_df], ignore_index=True)
                        results_df.to_csv(f"test_epoch{self.current_epoch}_eta.csv")

            results_df.reset_index(inplace=True)
            results_df.set_index(["num_step", "eta"], inplace=True)
            results_df.to_csv(f"test_epoch{self.current_epoch}_eta.csv")

        if self.cfg.general.search == 'conditional':  # cfg.general.search = None
            # self.cfg.sample.target_guided = True
            num_step_list = [1, 5, 10, 50, 100, 1000]

            results_df = pd.DataFrame()
            for num_step in num_step_list:
                # for guidance_weight in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
                for guidance_weight in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    # for guided_weight in [0.9, 1.1, 1.2, 1.3]:
                    self.cfg.sample.x1_parameterization = False
                    # self.cfg.sample.x1_parameterization = True
                    self.cfg.sample.time_distortion = 'polydec'
                    self.cfg.sample.omega = 1.0
                    self.cfg.sample.eta = 0.0
                    self.cfg.general.guidance_weight = guidance_weight

                    print(
                        f"############# Testing distortor: {guidance_weight}, num steps: {num_step} #############"
                    )
                    self.sample_T = num_step
                    res = self.sample_and_evaluate(
                        test=True, rdb="general", rdb_crit="general"
                    )
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["guidance_weight"] = guidance_weight
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(f"test_epoch{self.current_epoch}_conditional.csv")

            results_df.reset_index(inplace=True)
            results_df.set_index(["num_step", "guidance_weight"], inplace=True)
            results_df.to_csv(f"test_epoch{self.current_epoch}_conditional.csv")

        elif self.cfg.general.search == 'step':
            if self.cfg.dataset.name == "qm9":
                # best_omega = 0.05
                # best_omega = 0.1
                best_omega = 0.5
                best_eta = 0.0
                best_distortor = "polydec"
                num_step_list = [1, 5, 10, 30, 50, 100, 500]
            elif self.cfg.dataset.name == "guacamol":
                best_omega = 0.1
                best_eta = 100.0
                best_distortor = "polydec"
                num_step_list = [1, 5, 10, 30, 50, 100, 500]
            elif self.cfg.dataset.name == "moses":
                best_omega = 1.0
                best_eta = 100.0
                best_distortor = "polydec"
                num_step_list = [1, 5, 10, 30, 50, 100, 500]

            for i in range(5):
                for num_step in num_step_list:
                    if i == 0:
                        self.cfg.sample.x1_parameterization = True
                        self.cfg.sample.time_distortion = best_distortor
                        self.cfg.sample.omega = best_omega
                        self.cfg.sample.eta = best_eta
                        name = "best"
                        name = "step4"
                    elif i == 1:
                        self.cfg.sample.x1_parameterization = False
                        self.cfg.sample.time_distortion = best_distortor
                        self.cfg.sample.omega = best_omega
                        self.cfg.sample.eta = best_eta
                        name = "remove_distortor"
                        name = "step3"
                    elif i == 2:
                        self.cfg.sample.x1_parameterization = False
                        self.cfg.sample.time_distortion = best_distortor
                        self.cfg.sample.omega = best_omega
                        self.cfg.sample.eta = 0.0
                        name = "step2"
                    elif i == 3:
                        self.cfg.sample.x1_parameterization = False
                        self.cfg.sample.time_distortion = best_distortor
                        self.cfg.sample.omega = 0.0
                        name = "remove_omega"
                        self.cfg.sample.eta = 0
                        name = "step1"
                    else:
                        self.cfg.sample.x1_parameterization = False
                        self.cfg.sample.time_distortion = "identity"
                        self.cfg.sample.omega = 0
                        self.cfg.sample.eta = 0
                        name = "remove_eta"
                        name = "vanilla"
                    print(
                        f"############# Testing range: {name}, eta: {self.cfg.sample.eta}, omega: {self.cfg.sample.omega}, distortor: {self.cfg.sample.time_distortion}, num steps: {num_step} #############"
                    )
                    self.sample_T = num_step
                    res = self.sample_and_evaluate(
                        test=True, rdb="general", rdb_crit="general"
                    )
                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["num_step"] = num_step
                    res_df["omega"] = self.cfg.sample.omega
                    res_df["distortor"] = self.cfg.sample.time_distortion
                    res_df["eta"] = self.cfg.sample.eta
                    res_df["name"] = name
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(f"test_epoch{self.current_epoch}_step_opt.csv")

            results_df.reset_index(inplace=True)
            results_df.set_index(
                ["num_step", "omega", "distortor", "eta"], inplace=True
            )
            results_df.to_csv(f"test_epoch{self.current_epoch}_step_opt.csv")

        elif self.cfg.general.search == 'rdb':

            num_step_list = [1, 5, 10, 50, 100, 1000]
            if self.cfg.dataset.name == "qm9":
                num_step_list = [1, 5, 10, 50, 100, 500]
            if self.cfg.dataset.name == "guacamol":
                num_step_list = [1, 5, 10, 50, 100, 500]
            if self.cfg.dataset.name == "moses":
                num_step_list = [1, 5, 10, 50, 100, 500]

            num_step = num_step_list[-1]

            eta_list = [
                0.0, 5, 10, 25, 50, 100, 200,
            ]

            for eta in eta_list:
                for rdb_type in [('general', 'general'),
                                 ('column', 'max_marginal'),
                                 ('column', 'x_1'),
                                 ('column', 'p_xt_g_x1'),
                                 ('entry', 'first')]:
                    print(
                        f"############# Testing solver: {rdb_type}, num steps: {eta} #############"
                    )
                    self.sample_T = num_step
                    self.cfg.sample.eta = eta
                    res = self.sample_and_evaluate(
                        test=True, rdb=rdb_type[0], rdb_crit=rdb_type[1]
                    )

                    mean_res = {f"{key}_mean": res[key][0] for key in res}
                    std_res = {f"{key}_std": res[key][1] for key in res}
                    mean_res.update(std_res)
                    res_df = pd.DataFrame([mean_res])
                    res_df["rdb"] = f'{rdb_type[0]}_{rdb_type[1]}'
                    res_df["eta"] = eta
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(
                        f"test_epoch{self.current_epoch}_rdb.csv"
                    )

            results_df.reset_index(inplace=True)
            results_df.set_index(["rdb", "eta"], inplace=True)
            results_df.to_csv(f"test_epoch{self.current_epoch}_rdb.csv")

        else:
            # To recover
            to_log = self.sample_and_evaluate(test=True)

            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            self.print("Done testing.")

    def sample_and_evaluate(
            self, eta=None, rdb=None, rdb_crit=None, test=False, samples=None
    ):
        if eta is None:
            eta = self.cfg.sample.eta  # 0
        if rdb is None:
            rdb = self.cfg.sample.rdb  # 'general'
        if rdb_crit is None:
            rdb_crit = self.cfg.sample.rdb_crit  # "x_0"

        if test:
            samples_to_generate = (
                    self.cfg.general.final_model_samples_to_generate
                    * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = samples_to_generate

            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        if self.cfg.general.generated_path:  # NULL
            self.print("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
        elif samples is None:
            samples = []
            labels = []
            id = 0
            while samples_left_to_generate > 0:
                self.print(
                    f"Samples left to generate: {samples_left_to_generate}/"
                    f"{samples_to_generate}",
                    end="",
                    # flush=True,
                )
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                num_chain_steps = min(self.number_chain_steps, self.sample_T)
                cur_samples, cur_labels = self.sample_batch(
                    id,
                    to_generate,
                    num_nodes=None,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=num_chain_steps,
                    eta=eta,
                    rdb=rdb,
                    rdb_crit=rdb_crit,
                )
                samples.extend(cur_samples)
                labels.extend(cur_labels)
                id += to_generate
                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            """
            self.print("Saving the generated graphs")
            filename = f"generated_samples1.txt"
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"generated_samples{i}.txt"
                else:
                    break
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")
            self.print("Generated graphs Saved. Computing sampling metrics...")

            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)
            """

        self.print("Computing sampling metrics...")
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate

        to_log = {}
        if test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                              i * samples_to_evaluate: (i + 1) * samples_to_evaluate
                              ]
                cur_labels = labels[
                             i * samples_to_evaluate: (i + 1) * samples_to_evaluate
                             ]

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,  # 用不上了
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=test,
                    local_rank=self.local_rank,
                    labels=cur_labels if self.conditional else None,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_eta{eta}_{rdb_crit}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=test,
                local_rank=self.local_rank,
                labels=(
                    cur_labels
                    if (self.conditional and self.cfg.dataset.name in "tls")
                    else None
                ),
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, pz=None, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)  # 随机采样一批 bs，
        else:
            t_float = t
        t_int = torch.clamp((t_float * self.T).long().float() + 1, 1, self.T)  # 离散化后的整数时间步

        # sample random step 构建 xt的分布
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        # 给定x1、时间 t，构造出 xt的概率分布（其实就是做插值）
        prob_X_t, prob_E_t = self.p_xt_g_x1(X1=X_1_label, E1=E_1_label, t=t_float, pz=pz)

        # step 4 - sample noised data 从p(xt|x1)中采样一个xt
        sampled_t = diffusion_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    # compute_val_loss
    '''
    def compute_val_loss(
            self,
            pred,
            noisy_data,
            X,
            E,
            y,
            node_mask,
            test=False,
    ):
        """Computes an estimator for the variational lower bound.
         评估 验证损失，特别是在变分框架下估计 Evidence Lower Bound (ELBO) 的
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        t = noisy_data["t"]

        # Adjust dimensions to virtual classes
        pred.X, pred.E, pred.y = self.noise_dist.add_virtual_classes(
            pred.X, pred.E, pred.y
        )
        X, E, y = self.noise_dist.add_virtual_classes(X, E, y)

        # 1. 节点数量的 log 概率
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)  # 根据这个分布给出每个图节点数量N的 log probability

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(
            E * prob0.E.log()
        )

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        return nll
    '''

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()

        out = self.model(X, E, y, node_mask)
        return out  # utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)  即预测的X1，E1, y1

    @torch.no_grad()
    def sample_batch(
            self,
            batch_id: int,
            batch_size: int,
            keep_chain: int,
            number_chain_steps: int,
            save_final: int,
            num_nodes=None,
            eta: float = 0.0,
            rdb: str = "general",
            rdb_crit: str = "x_1",
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)  # bs，
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(  # 噪声 x0
            limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                # 随机选取 y 中的 100 个标签，避免总是用前几个样本，提高多样性。
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                # 将条件变量赋值给生成的起始 latent 向量 z_T 中的 .y 属性
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
                # 复制 condition 10 次（形成 1000 个条件），shape 会变成 (1000, d)，截取前 batch_size 个条件。
            else:
                raise NotImplementedError

        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        # chain_X,chain_E 相关
        """
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )
        
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        """

        for t_int in tqdm(range(0, self.sample_T)):  # test 1000，train 500
            # 生成过程 噪声x0->xt->xs->x1
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)  # 当前时间步 整数 bs个t_int
            t_norm = t_array / self.sample_T
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):  # 'marginal'
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.sample_T  # # 下一时间步
            if self.cfg.sample.time_schedule == "logitnormal":
                if t_int == 0:
                    t_norm = t_norm + 1e-5

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s  采样 z_s ~ p(z_s | z_t)（核心）
            # cur_eta = 0.0 if t_norm[0] < 0.95 else eta  # 控制重参数技巧的温度 / 噪声控制量（eta 越小，采样越 deterministic）
            cur_eta = eta
            # cur_eta = 0.0 if t_norm[0] > 0.5 else eta
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
                cur_eta,
                rdb,
                rdb_crit,
                # condition=condition,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y  # 把新采样的作为下一轮输入

            # Save the first keep_chain graphs
            """
            write_index = (t_int * number_chain_steps) // self.sample_T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            """

        # Sample  利用掩码 node_mask 去掉 padding  collapse=True应该是argmax了
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        """
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)
        """

        molecule_list = []
        label_list = []
        for i in range(batch_size):  # 处理每张图
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        # Visualize chains
        """    
        if self.visualization_tools is not None:
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path, chain_X[:, i, :].numpy(), chain_E[:, i, :].numpy()
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
        """

        # Visualize the final molecules
        """
            self.print("\nVisualizing graphs...")
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")
        """

        return molecule_list, label_list

    def dt_p_xt_g_x1(self, X1, E1):
        # x1 (B, D)
        # t float
        # returns (B, D, S) for varying x_t value
        # 计算一个从 已知的最终状态 $x_1$ 到 某个中间状态 $x_t$ 的“方向
        # ∇_下标{x_t} \log p_t(x_t | x_1) 的估计方向
        limit_dist = self.limit_dist.to_device(self.device)
        X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
        E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

        # 在分类变量里，log-likelihood 梯度方向可以近似为：
        # ∇_下标{x_t} \log p_t(x_t | x_1)  ≈ onehot(x1) -  limit_distribution
        # 这是 categorical log-likelihood 的梯度表现形式（见 softmax logit 梯度推导）
        dX = X1_onehot - limit_dist.X[None, None, :]  # (bs, n, dx)
        dE = E1_onehot - limit_dist.E[None, None, None, :]

        # 检查是否为 valid 概率差值
        # 每个 one-hot 和 limit_dist 差值的和应该是 0（因为 one-hot 的总和是1，limit_dist 总和也是1）
        assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

        # 估算给定 x1 下，某个 xt 的方向梯度项 ∇xt logp(xt ∣x1)
        return dX, dE

    def p_xt_g_x1(self, X1, E1, t, pz):
        # 在某一时间步 t下，如何从已知的终点（原始数据x1 ）构造出中间噪声状态 xt的分布
        # x1 (B, N) 每个值是类别索引  e1 (B, N, N)
        # t float (B, 1)
        # returns (B, N, S) for varying x_t value 概率分布

        # t_time = t.squeeze(-1)[:, None, None]

        pz = pz.to_device(self.device)

        # 处理 X1 和 E1 的 one-hot
        B, N = X1.shape
        dx = pz.X.shape[-1]
        de = pz.E.shape[-1]
        X1_onehot = F.one_hot(X1, num_classes=dx).float()  # (B, N, dx)
        E1_onehot = F.one_hot(E1, num_classes=de).float()  # (B, N, N, de)

        # 插值构造 xt
        # Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
        # Et = (
        #         t_time[:, None] * E1_onehot
        #         + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
        # )
        # 统一 limit_dist 到 batch 维度（如果是共享全局分布）
        limit_dist = copy.deepcopy(self.limit_dist).to_device(self.device)
        X_limit = limit_dist.X.view(1, 1, dx).expand(B, N, dx)  # (B, N, dx)
        E_limit = limit_dist.E.view(1, 1, 1, de).expand(B, N, N, de)  # (B, N, N, de)

        if pz.X.dim() == 2:
            # 如果 pz 是每个样本一个分布（B, dx），需要适配 batch
            k = min((self.current_epoch / 200) * 0.5, 0.5)
            # 渐进式混合先验
            X_limit = (1 - k) * X_limit + k * pz.X.unsqueeze(1).expand(B, N, dx)  # (B, N, dx)
            E_limit = (1 - k) * E_limit + k * pz.E.unsqueeze(1).unsqueeze(1).expand(B, N, N, de)  # (B, N, N, de)

        # t 插值（reshape 成 (B, 1, 1)）
        t_x = t.view(B, 1, 1)  # for X1_onehot
        t_e = t.view(B, 1, 1, 1)  # for E1_onehot

        Xt = t_x * X1_onehot + (1 - t_x) * X_limit
        Et = t_e * E1_onehot + (1 - t_e) * E_limit

        # 检验概率合法性  每个位置上的概率和是 1（soft one-hot 向量）
        assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
                (Et.sum(-1) - 1).abs() < 1e-4
        ).all()

        return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)

    def compute_pt_vals(self, t, X_t_label, E_t_label, X_1_pred, E_1_pred):
        # X_t_label 当前步 X_t 的 ground truth label，shape = (bs, n, 1)（取值范围是 [0, dx)）
        # E_t_label	当前步 E_t 的 label，shape = (bs, n, n, 1)

        # 表示在当前x1 下，某个 xt 的值变动会带来多大的变化率
        dt_p_vals_X, dt_p_vals_E = self.dt_p_xt_g_x1(
            X_1_pred, E_1_pred
        )  # (bs, n, dx), (bs, n, n, de)  梯度方向估计

        # 提取当前 x_t 对应的那一类维度值  ∂t pt(xt=a∣x1 )
        dt_p_vals_at_Xt = dt_p_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        pt_vals_X, pt_vals_E = self.p_xt_g_x1(  # 在给定 x_1 的前提下，各类 x_t 的概率分布
            # X_1_pred, E_1_pred, t + dt
            X_1_pred,
            E_1_pred,
            t,
            self.limit_dist
        )  # (bs, n, dx), (bs, n, n, de)

        # 获取当前 label 的概率  提取出当前 ground truth X_t_label 对应的概率值
        pt_vals_at_Xt = pt_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        # 变量	                     含义	              用于后续计算什么
        # pt_vals_X / pt_vals_E	    所有类型的概率向量	      构造 full rate
        # pt_vals_at_Xt	            当前 label 的概率值	  构造 $R_t^*$ 的分母
        # dt_p_vals_X / dt_p_vals_E	所有类型的导数	      构造 full vector field
        # dt_p_vals_at_Xt	        当前 label 的导数值	  构造 $R_t^*$ 的分子
        return (
            pt_vals_X,
            pt_vals_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            dt_p_vals_X,
            dt_p_vals_E,
            dt_p_vals_at_Xt,
            dt_p_vals_at_Et,
        )

    def compute_Rstar(
            self,
            X_1_pred,
            E_1_pred,
            X_t_label,
            E_t_label,
            pt_vals_X,
            pt_vals_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            dt_p_vals_X,
            dt_p_vals_E,
            dt_p_vals_at_Xt,
            dt_p_vals_at_Et,
            func,
    ):
        # Numerator of R_t^*
        # R* 权重计算函数，在 训练过程中计算 R*_t(x) 的一个关键组成部分，用于将噪声点x_0 映射到数据点x_1 的“最优向量场”
        inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]

        # compensate
        limit_dist = self.limit_dist.to_device(self.device)
        X1_onehot = F.one_hot(X_1_pred, num_classes=len(limit_dist.X)).float()  # 预测的
        E1_onehot = F.one_hot(E_1_pred, num_classes=len(limit_dist.E)).float()
        mask_X = X_1_pred.unsqueeze(-1) != X_t_label
        mask_E = E_1_pred.unsqueeze(-1) != E_t_label

        Rstar_t_numer_X = F.relu(inner_X)  # (bs, n, dx)
        Rstar_t_numer_E = F.relu(inner_E)  # (bs, n, n, de)

        # target guidance scheme 2
        Rstar_t_numer_X += X1_onehot * self.cfg.sample.omega * mask_X  # self.cfg.sample.omega：0
        Rstar_t_numer_E += E1_onehot * self.cfg.sample.omega * mask_E

        Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

        # Denominator of R_t^*
        Rstar_t_denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        Rstar_t_denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)
        Rstar_t_X = Rstar_t_numer_X / Rstar_t_denom_X[:, :, None]  # (bs, n, dx)
        Rstar_t_E = Rstar_t_numer_E / Rstar_t_denom_E[:, :, :, None]  # (B, n, n, de)

        Rstar_t_X = torch.nan_to_num(Rstar_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        Rstar_t_E = torch.nan_to_num(Rstar_t_E, nan=0.0, posinf=0.0, neginf=0.0)

        Rstar_t_X[Rstar_t_X > 1e5] = 0.0
        Rstar_t_E[Rstar_t_E > 1e5] = 0.0

        return Rstar_t_X, Rstar_t_E

    def compute_RDB(
            self,
            pt_vals_X,
            pt_vals_E,
            X_t_label,
            E_t_label,
            pred_X,
            pred_E,
            X_1_pred,
            E_1_pred,
            rdb,  # 'general'
            rdb_crit,
            node_mask,
            t,
    ):
        dx = pt_vals_X.shape[-1]
        de = pt_vals_E.shape[-1]
        # Masking Rdb
        if rdb == "general":
            x_mask = torch.ones_like(pt_vals_X)
            e_mask = torch.ones_like(pt_vals_E)
        elif rdb == "marginal":
            x_limit = self.limit_dist.X
            e_limit = self.limit_dist.E

            Xt_marginal = x_limit[X_t_label]
            Et_marginal = e_limit[E_t_label]

            x_mask = x_limit.repeat(X_t_label.shape[0], X_t_label.shape[1], 1)
            e_mask = e_limit.repeat(
                E_t_label.shape[0], E_t_label.shape[1], E_t_label.shape[2], 1
            )

            x_mask = x_mask > Xt_marginal
            e_mask = e_mask > Et_marginal

        elif rdb == "column":
            # Get column idx to pick
            if rdb_crit == "max_marginal":
                x_column_idxs = (
                    self.noise_dist.get_limit_dist()
                    .X.argmax(keepdim=True)
                    .expand(X_t_label.shape)
                )
                e_column_idxs = (
                    self.noise_dist.get_limit_dist()
                    .E.argmax(keepdim=True)
                    .expand(E_t_label.shape)
                )
            elif rdb_crit == "x_t":
                x_column_idxs = X_t_label
                e_column_idxs = E_t_label
            elif rdb_crit == "abs_state":
                x_column_idxs = torch.ones_like(X_t_label) * (dx - 1)
                e_column_idxs = torch.ones_like(E_t_label) * (de - 1)
            elif rdb_crit == "p_x1_g_xt":
                x_column_idxs = pred_X.argmax(dim=-1, keepdim=True)
                e_column_idxs = pred_E.argmax(dim=-1, keepdim=True)
            elif rdb_crit == "x_1":  # as in paper, uniform
                x_column_idxs = X_1_pred.unsqueeze(-1)
                e_column_idxs = E_1_pred.unsqueeze(-1)
            elif rdb_crit == "p_xt_g_x1":
                x_column_idxs = pt_vals_X.argmax(dim=-1, keepdim=True)
                e_column_idxs = pt_vals_E.argmax(dim=-1, keepdim=True)
            elif rdb_crit == "xhat_t":
                sampled_1_hat = diffusion_utils.sample_discrete_features(
                    pt_vals_X,
                    pt_vals_E,
                    node_mask=node_mask,
                )
                x_column_idxs = sampled_1_hat.X.unsqueeze(-1)
                e_column_idxs = sampled_1_hat.E.unsqueeze(-1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            x_mask = F.one_hot(x_column_idxs.squeeze(-1), num_classes=dx)
            x_mask[(x_column_idxs == X_t_label).squeeze(-1)] = 1.0
            e_mask = F.one_hot(e_column_idxs.squeeze(-1), num_classes=de)
            e_mask[(e_column_idxs == E_t_label).squeeze(-1)] = 1.0

        elif rdb == "entry":
            if rdb_crit == "abs_state":
                # select last index
                x_masked_idx = torch.tensor(
                    dx
                    - 1  # delete -1 for the last index
                    # dx - 1
                ).to(
                    self.device
                )  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(de - 1).to(self.device)

                x1_idxs = X_1_pred.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_pred.unsqueeze(-1)  # (bs, n, n, 1)
            if rdb_crit == "first":  # here in all datasets it's the argmax
                # select last index
                x_masked_idx = torch.tensor(0).to(
                    self.device
                )  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(0).to(self.device)

                x1_idxs = X_1_pred.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_pred.unsqueeze(-1)  # (bs, n, n, 1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            # bs, n, _ = X_t_label.shape
            # x_mask = torch.zeros((bs, n, dx), device=self.device)  # (bs, n, dx)
            x_mask = torch.zeros_like(pt_vals_X)  # (bs, n, dx)
            xt_in_x1 = (X_t_label == x1_idxs).squeeze(-1)  # (bs, n, 1)
            x_mask[xt_in_x1] = F.one_hot(x_masked_idx, num_classes=dx).float()
            xt_in_masked = (X_t_label == x_masked_idx).squeeze(-1)
            x_mask[xt_in_masked] = F.one_hot(
                x1_idxs.squeeze(-1), num_classes=dx
            ).float()[xt_in_masked]

            # e_mask = torch.zeros((bs, n, n, de), device=self.device)  # (bs, n, dx)
            e_mask = torch.zeros_like(pt_vals_E)
            et_in_e1 = (E_t_label == e1_idxs).squeeze(-1)
            e_mask[et_in_e1] = F.one_hot(e_masked_idx, num_classes=de).float()
            et_in_masked = (E_t_label == e_masked_idx).squeeze(-1)
            e_mask[et_in_masked] = F.one_hot(
                e1_idxs.squeeze(-1), num_classes=de
            ).float()[et_in_masked]
        else:
            raise NotImplementedError

        return x_mask, e_mask

    def compute_R(
            self,
            Rstar_t_X,
            Rstar_t_E,
            Rdb_t_X,
            Rdb_t_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            pt_vals_X,
            pt_vals_E,
            node_mask,
    ):
        # sum to get the final R_t_X and R_t_E
        R_t_X = Rstar_t_X + Rdb_t_X
        R_t_E = Rstar_t_E + Rdb_t_E

        # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero, which need to be applied to Rdb too
        dx = R_t_X.shape[-1]
        de = R_t_E.shape[-1]
        R_t_X[(pt_vals_at_Xt == 0.0)[:, :, None].repeat(1, 1, dx)] = 0.0
        R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].repeat(1, 1, 1, de)] = 0.0
        # zero-out certain columns of R, which is implied in the computation of Rdb
        # if the probability of a place is 0, then we should not consider it in the R computation
        R_t_X[pt_vals_X == 0.0] = 0.0
        R_t_E[pt_vals_E == 0.0] = 0.0

        return R_t_X, R_t_E

    def compute_rate_matrix(
            self,
            t,
            eta,
            rdb,
            rdb_crit,
            X_1_pred,
            E_1_pred,
            X_t_label,
            E_t_label,
            pred_X,
            pred_E,
            node_mask,
            pc_dt=0,
            return_rdb=False,
            return_rstar=False,
            return_both=False,
            func="relu",
    ):  # 构造 rate matrix（速率矩阵）
        (  # 构造真实目标场Rt∗ 所需信息
            pt_vals_X,
            pt_vals_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            dt_p_vals_X,
            dt_p_vals_E,
            dt_p_vals_at_Xt,
            dt_p_vals_at_Et,
        ) = self.compute_pt_vals(t, X_t_label, E_t_label, X_1_pred, E_1_pred)
        # ) = self.compute_pt_vals(t, X_column_to_keep, E_column_to_keep, X_1_pred, E_1_pred)

        Rstar_t_X, Rstar_t_E = self.compute_Rstar(  # 构造目标向量场 Rt∗
            X_1_pred,
            E_1_pred,
            X_t_label,
            E_t_label,
            pt_vals_X,
            pt_vals_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            dt_p_vals_X,
            dt_p_vals_E,
            dt_p_vals_at_Xt,
            dt_p_vals_at_Et,
            func,
        )

        X_mask, E_mask = self.compute_RDB(  # 计算 mask（可训练控制 mask）——RDB
            # 核心是构造一个 rate diffusion bridge（RDB）路径上的控制 mask。
            # 衡量当前的预测是否可信，目的是：
            # 如果某个 $x_t$ 的概率太低（即不确定） → 屏蔽该维度
            # 如果预测很差（偏离目标很多） → 加强该维度
            # 它用于控制 rate matrix 的方向性
            pt_vals_X,
            pt_vals_E,
            X_t_label,
            E_t_label,
            pred_X,
            pred_E,
            X_1_pred,
            E_1_pred,
            rdb,
            rdb_crit,
            node_mask,
            t,
        )

        # stochastic rate matrix 构造 Rdb_t，基于 Mask 的“数据引导的向量场”
        Rdb_t_X = pt_vals_X * X_mask * eta
        Rdb_t_E = pt_vals_E * E_mask * eta

        R_t_X, R_t_E = self.compute_R(  # 得到最终速率 Rt
            Rstar_t_X,
            Rstar_t_E,
            Rdb_t_X,
            Rdb_t_E,
            pt_vals_at_Xt,
            pt_vals_at_Et,
            pt_vals_X,
            pt_vals_E,
            node_mask,
        )

        if return_rstar:
            return R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, X_mask, E_mask

        if return_rdb:
            return R_t_X, R_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask

        if return_both:
            return R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask

        return R_t_X, R_t_E, X_mask, E_mask

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def compute_graph_rate_matrix(
            self,
            pred_X,
            pred_E,
            X_t,
            E_t,
            dt,
            limit_x,
            limit_e,
            node_mask,
            t,
            eta,
            rdb,
            rdb_crit,
            y_t,
            # condition,
    ):
        # Zero-out non-existing states
        dx = limit_x.shape[-1]
        de = limit_e.shape[-1]

        X_t_label = X_t.argmax(-1, keepdim=True)
        E_t_label = E_t.argmax(-1, keepdim=True)
        if not self.cfg.sample.x1_parameterization:
            sampled_1 = diffusion_utils.sample_discrete_features(
                pred_X, pred_E, node_mask=node_mask
            )
            X_1_pred = sampled_1.X
            E_1_pred = sampled_1.E

            pc_dt = dt * self.cfg.sample.guided_weight
            R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask = (
                self.compute_rate_matrix(
                    t,
                    eta,
                    rdb,
                    rdb_crit,
                    X_1_pred,
                    E_1_pred,
                    X_t_label,
                    E_t_label,
                    pred_X,
                    pred_E,
                    node_mask,
                    return_both=True,
                    pc_dt=pc_dt,
                )
            )

        else:
            bs, n, dx = X_t.shape
            dummy_X_1_pred = torch.zeros((bs, n)).long().to(self.device)
            dummy_E_1_pred = torch.zeros((bs, n, n)).long().to(self.device)
            # Built R_t_X
            R_t_X_list = []

            for x_1 in range(dx):
                X_1_pred = x_1 * torch.ones_like(dummy_X_1_pred).long().to(
                    self.device
                )
                R_t_X, R_t_E, X_mask, E_mask = self.compute_rate_matrix(
                    t,
                    eta,
                    rdb,
                    rdb_crit,
                    X_1_pred,
                    dummy_E_1_pred,
                    X_t_label,
                    E_t_label,
                    pred_X,
                    pred_E,
                    node_mask,
                )
                R_t_X_list.append(R_t_X)

            R_t_X_stacked = torch.stack(R_t_X_list, dim=-1)
            R_t_X = torch.sum(
                R_t_X_stacked * pred_X.unsqueeze(-2), dim=-1
            )  # weight sum

            # Built R_t_E
            R_t_E_list = []
            for e_1 in range(de):
                E_1_pred = e_1 * torch.ones_like(dummy_E_1_pred).long().to(
                    self.device
                )
                R_t_X, R_t_E, X_mask, E_mask = self.compute_rate_matrix(
                    t,
                    eta,
                    rdb,
                    rdb_crit,
                    dummy_X_1_pred,
                    E_1_pred,
                    X_t_label,
                    E_t_label,
                    pred_X,
                    pred_E,
                    node_mask,
                )
                R_t_E_list.append(R_t_E)
            R_t_E_stacked = torch.stack(R_t_E_list, dim=-1)

            R_t_E = torch.sum(
                R_t_E_stacked * pred_E.unsqueeze(-2), dim=-1
            )  # weight sum

        return R_t_X, R_t_E

    def sample_p_zs_given_zt(
            self,
            t,
            s,
            X_t,
            E_t,
            y_t,
            node_mask,
            eta,
            rdb,
            rdb_crit,
            # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        # 计算 rate（速度）矩阵  构造从当前状态 zt 向未来状态zs 的转移率（rate matrix），就是 Rt(xt)
        R_t_X, R_t_E = self.compute_graph_rate_matrix(
            pred_X,  # 概率
            pred_E,
            X_t,
            E_t,
            dt,  # 时间步长
            limit_x,
            limit_e,
            node_mask,
            t,  # 当前时间
            eta,
            rdb,
            rdb_crit,
            y_t,  # y_t 是当前 batch 的条件变量
            # y_t=torch.ones_like(y_t, device=self.device) * -1,
            # condition=None,
        )

        if self.conditional:
            # Classifier-Free Guidance 的实现
            # 将 无条件预测 (uncond) 和 有条件预测 (cond) 的结果融合起来，引导生成过程更加符合目标属性
            uncond_y = torch.ones_like(y_t, device=self.device) * -1  # uncond_y = -1 表示 不使用任何条件
            noisy_data["y_t"] = uncond_y
            # noisy_data是 xt

            # 计算无条件预测
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            # 计算 rate matrix（扩散速率）
            R_t_X_uncond, R_t_E_uncond = self.compute_graph_rate_matrix(
                pred_X,
                pred_E,
                X_t,
                E_t,
                dt,
                limit_x,
                limit_e,
                node_mask,
                t,
                eta,
                rdb,
                rdb_crit,
                # y_t,
                uncond_y,  # # 注意：这里用的是无条件版本
                # condition=condition,
            )

            # 融合无条件和有条件结果（Classifier-Free Guidance）
            guidance_weight = self.cfg.general.guidance_weight  # 2
            R_t_X = torch.exp(  # 用 log-space 中的线性插值 合并有/无条件两个速率 ，经典的 Classifier-Free 融合方式
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        # 根据速率采样一步转移概率 p(zs|zt)
        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:  # 如果是最后一步
            prob_X, prob_E = pred_X, pred_E  # 如果已经到终点（时间 1），直接用最终的 softmax 作为概率分布

        try:  # 从多项分布中 sample 一个 zs
            sampled_s = diffusion_utils.sample_discrete_features(
                prob_X, prob_E, node_mask=node_mask
            )
        except:
            import pdb
            pdb.set_trace()

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
