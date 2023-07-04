import os
import sys
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import List

from src.models.model import Model
from src.models.FSL.IPN.weight_module import Weight
from src.models.FSL.IPN.distance_module import DistScale
from src.models.FSL.ProtoNet.proto_batch_sampler import PrototypicalBatchSampler
from src.models.FSL.ProtoNet.proto_loss import prototypical_loss as loss_fn
from src.models.FSL.ProtoNet.proto_loss import ProtoTools, TestResult
from src.utils.tools import Tools, Logger
from src.utils.config_parser import Config
from src.datasets.staple_dataset import CustomDataset
from src.train_test.routine import TrainTest
from config.consts import General as _CG
from config.consts import SubsetsDict


class ProtoRoutine(TrainTest):

    def __init__(self, model: Model, dataset: CustomDataset, subsets_dict: SubsetsDict):
        super().__init__(model, dataset, subsets_dict)
        self.learning_rate = 0.001
        self.lr_scheduler_gamma = 0.5
        self.lr_scheduler_step = 20

    def init_loader(self, config: Config, split_set: str):
        current_subset = self.get_subset_info(split_set)
        
        if current_subset.subset is None:
            return None
        
        min_req = config.fsl.test_k_shot_q + config.fsl.train_k_shot_s
        if any(map(lambda x: x < min_req, current_subset.info_dict.values())):
            raise ValueError(f"at least one class has not enough elements {(min_req)}. Check {current_subset.info_dict}")
        
        label_list = [self.dataset[idx][1] for idx in current_subset.subset.indices]
        sampler = PrototypicalBatchSampler(
            label_list,
            config.fsl.train_n_way if split_set == self.train_str else config.fsl.test_n_way,
            config.fsl.train_k_shot_s + config.fsl.train_k_shot_q if split_set == self.train_str else config.fsl.test_k_shot_s + config.fsl.test_k_shot_q,
            config.fsl.episodes
        )
        return DataLoader(current_subset.subset, batch_sampler=sampler)

    def train(self, config: Config):
        Logger.instance().debug("Start training")
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        trainloader = self.init_loader(config, self.train_str)
        valloader = self.init_loader(config, self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.lr_scheduler_gamma,
            step_size=self.lr_scheduler_step
        )

        ## extra modules
        n_way, k_support, k_query = (config.fsl.train_n_way, config.fsl.train_k_shot_s, config.fsl.train_k_shot_q)
        dummy_batch = torch.Tensor(next(iter(trainloader))[0]).detach().to(_CG.DEVICE)
        proto_out_size = Model.get_output_size(self.model, dummy_batch)
        del dummy_batch
        
        # weight module
        weight_module = Weight(proto_out_size * k_support, k_support).to(_CG.DEVICE)

        # distance scale module
        dist_module = DistScale(proto_out_size * k_query * (k_support + k_query), k_support).to(_CG.DEVICE)
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0
        best_loss = float("inf")

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        best_model_path = os.path.join(out_folder, "best_model.pth")
        val_model_path = os.path.join(out_folder, "val_model.pth")
        last_model_path = os.path.join(out_folder, "last_model.pth")
        last_val_model_path = os.path.join(out_folder, "last_val_model.pth")

        for eidx, epoch in enumerate(range(config.epochs)):
            Logger.instance().debug(f"=== Epoch: {epoch} ===")
            self.model.train()
            for x, y in tqdm(trainloader):
                optim.zero_grad()
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                
                s_batch, q_batch = ProtoTools.split_support_query(model_output, y, n_way, k_support, k_query)
                prototypes = weight_module(s_batch.view(s_batch.shape[0], -1))
                s_cat_q = DistScale.cat_support_query(s_batch, q_batch)
                alphas = dist_module(s_cat_q.view(s_cat_q.shape[0], -1))
                
                loss, acc = ProtoTools.proto_loss(alphas, q_batch, prototypes)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            
            avg_loss = np.mean(train_loss[-config.fsl.episodes:])
            avg_acc = np.mean(train_acc[-config.fsl.episodes:])
            lr_scheduler.step()
            
            Logger.instance().debug(f"Avg Train Loss: {avg_loss}, Avg Train Acc: {avg_acc}")

            # save model
            if avg_acc >= best_acc:
                Logger.instance().debug(f"Found the best model at epoch {epoch}!")
                best_acc = avg_acc
                torch.save(self.model.state_dict(), best_model_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # wandb
            wdb_dict = { "train_loss": avg_loss, "train_acc": avg_acc }

            ## VALIDATION
            if valloader is not None:
                avg_loss_eval, avg_acc_eval = self.validate(config, valloader, val_loss, val_acc)
                if avg_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), val_model_path)

                # wandb
                wdb_dict["val_loss"] = avg_loss_eval
                wdb_dict["val_acc"] = avg_acc_eval    
            ## EOF: VALIDATION
            
            # wandb
            wandb.log(wdb_dict)

            # stop conditions and save last model
            if eidx == config.epochs-1 or best_acc >= 1.0-_CG.EPS_ACC or best_loss <= 0.0+_CG.EPS_LSS:
                pth_path = last_val_model_path if valloader is not None else last_model_path
                Logger.instance().debug(f"STOP: saving last epoch model named `{os.path.basename(pth_path)}`")
                torch.save(self.model.state_dict(), pth_path)

                # wandb: save all models
                wandb.save(f"{out_folder}/*.pth")

                return

    def validate(self, config: Config, valloader: DataLoader, val_loss: List[float], val_acc: List[float]):
        Logger.instance().debug("Validating!")
        
        self.model.eval()
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                loss, acc = loss_fn(model_output, target=y, n_support=config.fsl.test_k_shot_s)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            avg_loss_eval = np.mean(val_loss[-config.fsl.episodes:])
            avg_acc_eval = np.mean(val_acc[-config.fsl.episodes:])

        Logger.instance().debug(f"Avg Val Loss: {avg_loss_eval}, Avg Val Acc: {avg_acc_eval}")

        return avg_loss_eval, avg_acc_eval

    def test(self, config: Config, model_path: str):
        Logger.instance().debug("Start testing")
        
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")
        
        try:
            model_path = Tools.validate_path(model_path)
            testloader = self.init_loader(config, self.test_str)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)
        except ValueError as ve:
            Logger.instance().error(f"{ve.args}. No test performed")
            return

        self.model.load_state_dict(torch.load(model_path))
        
        legacy_avg_acc = list()
        acc_per_epoch = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.info_dict.keys())) }

        tr_acc_max = 0.0
        tr_max = TestResult()
        
        self.model.eval()
        with torch.no_grad():
            for epoch in tqdm(range(10)):
                tr = TestResult()
                score_per_class = { i: torch.FloatTensor().to(_CG.DEVICE) for i in range(len(self.test_info.info_dict.keys())) }
                for x, y in testloader:
                    x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                    y_pred = self.model(x)

                    # (overall accuracy [legacy], accuracy per class)
                    legacy_acc, acc_vals = tr.proto_test(y_pred, target=y, n_support=config.fsl.test_k_shot_s)
                    legacy_avg_acc.append(legacy_acc.item())
                    for k, v in acc_vals.items():
                        score_per_class[k] = torch.cat((score_per_class[k], v.reshape(1,)))
                
                avg_score_class = { k: torch.mean(v) for k, v in score_per_class.items() }
                Logger.instance().debug(f"at epoch {epoch}, average test accuracy: {avg_score_class}")

                for k, v in avg_score_class.items():
                    acc_per_epoch[k] = torch.cat((acc_per_epoch[k], v.reshape(1,)))

                tr.acc_overall = tr.acc_overall.mean()
                if tr.acc_overall > tr_acc_max:
                    tr_max = tr

        avg_acc_epoch = { k: torch.mean(v) for k, v in acc_per_epoch.items() }
        Logger.instance().debug(f"Accuracy on epochs: {avg_acc_epoch}")
        
        legacy_avg_acc = np.mean(legacy_avg_acc)
        Logger.instance().debug(f"Legacy test accuracy: {legacy_avg_acc}")

        if config.fsl.test_n_way == len(self.dataset.label_to_idx.keys()) and len(tr_max.target_inds) != 0 and len(tr_max.y_hat) != 0:
            y_true = tr_max.target_inds
            preds = tr_max.y_hat
            wandb.log({
                "confusion": wandb.plot.confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    preds=preds.cpu().detach().numpy(),
                    class_names=list(self.dataset.label_to_idx.keys())
                    )
                })
