import os
import sys
import torch
import wandb

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.datasets.staple_dataset import CustomDataset
from config.consts import General as _CG
from config.consts import SubsetsDict
from src.models.model import Model
from src.train_test.routine import TrainTest
from src.utils.config_parser import Config
from src.utils.tools import Logger, Tools


class StandardRoutine(TrainTest):

    def __init__(self, model: Model, dataset: CustomDataset, subsets_dict: SubsetsDict):
        super().__init__(model, dataset, subsets_dict)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(_CG.DEVICE)

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
        top_pred = y_pred.argmax(1, keepdim=True)           # select the max class (the one with the highest score)
        correct = top_pred.eq(y.view_as(top_pred)).sum()    # count the number of correct predictions
        return correct.float() / y.shape[0]                 # compute percentage of correct predictions (accuracy score)
    
    def init_loader(self, config: Config, split_set: str):
        current_subset = self.get_subset_info(split_set)
        if current_subset.subset is None:
            return None
        
        return DataLoader(current_subset.subset, batch_size=config.batch_size)

    def train(self, config: Config):
        Logger.instance().debug("Start training")
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        trainloader = self.init_loader(config, self.train_str)
        valloader = self.init_loader(config, self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        
        # wandb
        example_data, examples_target = next(iter(trainloader))
        img_grid = make_grid(example_data)
        wandb.log({"examples": wandb.Image(img_grid, caption="Top: Output, Bottom: Input")})

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        best_model_path = os.path.join(out_folder, "best_model.pth")
        val_model_path = os.path.join(out_folder, "val_model.pth")
        last_model_path = os.path.join(out_folder, "last_model.pth")
        last_val_model_path = os.path.join(out_folder, "last_val_model.pth")
        
        best_loss = float('inf')
        best_acc = 0

        for eidx, epoch in enumerate(range(config.epochs)):
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0

            for image, label in tqdm(trainloader, desc="Training", leave=False): 
                optim.zero_grad()
                image = image.to(_CG.DEVICE) #.reshape(-1, shape, shape) # alternative x.view(x.shape[0], -1) -> [batch, dim*dim]
                label = label.to(_CG.DEVICE)

                # forward pass
                pred = self.model(image)                        # [batch_size, n_classes]
                loss = self.criterion(pred, label)
                acc = StandardRoutine.compute_accuracy(pred.data, label)

                # backward and optimize
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_loss = epoch_loss / len(trainloader)
            epoch_acc = epoch_acc / len(trainloader)

            if epoch_acc > best_acc and epoch > 0:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), best_model_path)
                Logger.instance().debug(f"saving model at iteration {epoch}.")

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            # wandb
            wdb_dict = { "train_loss": epoch_loss, "train_acc": epoch_acc }

            ## VALIDATION
            if valloader is not None:
                epoch_loss_eval, epoch_acc_eval = self.validate(config, valloader)
                if epoch_acc_eval >= best_acc:
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), val_model_path)

                # wandb
                wdb_dict["val_loss"] = epoch_loss_eval
                wdb_dict["val_acc"] = epoch_acc_eval
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

    def validate(self, config: Config, valloader: DataLoader):
        Logger.instance().debug("Validating!")
        self.model.eval()
        
        tot_samples = 0
        tot_correct = 0

        loss = 0
        
        for images, labels in valloader:
            images = images.to(_CG.DEVICE)
            labels = labels.to(_CG.DEVICE)
            
            y_pred = self.model(images)
            loss = self.criterion(y_pred, labels)
            
            # max returns (value, index)
            top_pred_val, top_pred_idx = torch.max(y_pred.data, 1)
            n_correct = top_pred_idx.eq(labels.view_as(top_pred_idx)).sum()
            
            # accuracy
            tot_samples += labels.size(0)
            tot_correct += n_correct

        acc = tot_correct / tot_samples

        Logger.instance().debug(f"Avg Val Loss: {loss}, Avg Val Acc: {acc}")

        return loss, acc
    
    def test(self, config: Config, model_path: str):
        Logger.instance().debug("Start testing")
        
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")
        
        try:
            model_path = Tools.validate_path(model_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)

        self.model.load_state_dict(torch.load(model_path))
        testloader = self.init_loader(config, self.test_str)
        
        self.model.eval()
        with torch.no_grad():
            tot_samples = 0
            tot_correct = 0
            for images, labels in testloader:
                images = images.to(_CG.DEVICE)
                labels = labels.to(_CG.DEVICE)
                
                y_pred = self.model(images)
                
                # max returns (value, index)
                top_pred_val, top_pred_idx = torch.max(y_pred.data, 1)
                n_correct = top_pred_idx.eq(labels.view_as(top_pred_idx)).sum()
                
                # accuracy
                tot_samples += labels.size(0)
                tot_correct += n_correct

                # wandb
                wandb.log({
                    "pr_curve": wandb.plot.pr_curve(
                        labels.cpu().detach().numpy(),
                        y_pred.cpu().detach().numpy(),
                        labels=list(self.dataset.label_to_idx.keys())
                        )
                    })
                wandb.log({
                    "confusion": wandb.plot.confusion_matrix(
                        y_true=labels.cpu().detach().numpy(),
                        preds=y_pred.cpu().detach().numpy(),
                        class_names=list(self.dataset.label_to_idx.keys())
                        )
                    })

            acc = tot_correct / tot_samples
            Logger.instance().debug(f"Test accuracy on {len(self.test_info.subset.indices)} images: {acc:.3f}")