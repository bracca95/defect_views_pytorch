import torch

from typing import Optional, Tuple
from torch.nn import functional as F

from config.consts import General as _CG


class ProtoTools:
    
    @staticmethod
    def euclidean_dist(x, y):
        """Compute euclidean distance between two tensors
        
        Args:
            x (torch.Tensor) of size N x D
            y (torch.Tensor) of size M x D
        """
        
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    @staticmethod
    def split_support_query(recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor]:
        classes = torch.unique(target)
        if not n_way == len(classes):
            raise ValueError(f"number of unique classes ({len(classes)}) must match config n_way ({n_way})")
        
        if not target.shape[0] // len(classes) == n_support + n_query:
            raise ValueError(f"target shape ({target.shape[0]}) does not match support ({n_support}) + query ({n_query})")
        
        class_idx = torch.stack(list(map(lambda x: torch.where(target == x)[0], classes)))  # shape = (n_way, s+q)
        support_idxs, query_idxs = torch.split(class_idx, [n_support, n_query], dim=1)

        support_set = recons[support_idxs.flatten()].view(n_way, n_support, -1)
        query_set = recons[query_idxs.flatten()].view(n_way, n_query, -1)

        return support_set, query_set
    
    @staticmethod
    def get_mean_dist(s_batch: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        n_classes, _, n_feat = (s_batch.shape)
        
        proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos)
        mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)

        return mean_dists
    
    @staticmethod
    def proto_loss(alphas: Optional[torch.Tensor], q_batch: torch.Tensor, protos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_classes, n_query, n_feat = (q_batch.shape)
        
        dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos)
        dists = dists if alphas is None else alphas * dists

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes).to(_CG.DEVICE)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val, acc_val


class TestResult:
    def __init__(self):
        self.acc_overall = torch.Tensor().to(_CG.DEVICE)
        self.y_hat = torch.Tensor().to(_CG.DEVICE)
        self.target_inds = torch.Tensor().to(_CG.DEVICE)

    def proto_test(self, alphas: Optional[torch.Tensor], q_batch: torch.Tensor, protos: torch.Tensor):
        n_classes, n_query, n_feat = (q_batch.shape)
        mapping = {i: i for i in range(n_classes)}
        
        dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos)
        dists = dists if alphas is None else alphas * dists

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes).to(_CG.DEVICE)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        _, y_hat = log_p_y.max(2)
        acc_overall = y_hat.eq(target_inds.squeeze(2)).float().mean()
        acc_vals = { c: y_hat[c].eq(target_inds.squeeze(2)[c]).float().mean() for c in range(n_classes) }

        self.acc_overall = torch.cat((self.acc_overall, acc_overall.flatten()))
        self.y_hat = torch.cat((self.y_hat, y_hat.flatten()))
        self.target_inds = torch.cat((self.target_inds, target_inds.flatten()))

        return acc_overall, { v: acc_vals[i] for i, v in enumerate(mapping.values()) }
