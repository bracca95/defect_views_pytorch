import torch

from typing import Optional, Tuple
from torch import nn
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

        return torch.sqrt(torch.pow(x - y, 2).sum(2))

    @staticmethod
    def split_support_query(recons: torch.Tensor, target: torch.Tensor, n_way: int, n_support: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # check correct input
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
    def proto_loss(s_batch: torch.Tensor, q_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_classes, n_query, n_feat = (q_batch.shape)

        protos = torch.mean(s_batch, dim=1)

        proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos)
        query_dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos)
        mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
        
        # proto vs query
        proto_exp = proto_dists.unsqueeze(0).expand(query_dists.size(0), -1, -1)
        query_exp = query_dists.unsqueeze(1).expand(-1, proto_dists.size(0), -1)
        #sim = torch.nn.CosineSimilarity(dim=2)(proto_exp, query_exp)
        dis = torch.sqrt(torch.pow(proto_exp - query_exp, 2).sum(2))
        # #idx = torch.argmax(sim, dim=-1)
        # #idx_class = torch.div(idx, n_query, rounding_mode='floor')
        
        # # mean vs query
        # mean_exp = mean_dists.unsqueeze(0).expand(query_dists.size(0), -1, -1)
        # query_exp = query_dists.unsqueeze(1).expand(-1, mean_dists.size(0), -1)
        # #sim = torch.nn.CosineSimilarity(dim=2)(mean_exp, query_exp)
        # dis = torch.sqrt(torch.pow(mean_exp - query_exp, 2).sum(2))

        log_p_y = F.log_softmax(-dis, dim=1).view(n_classes, n_query, -1)
        #log_p_y = idx_class.view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes).to(_CG.DEVICE)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, idx_class = log_p_y.max(2)   # idx = torch.argmax(sim, dim=-1)
        
        # proto vs query
        y_hat = torch.div(idx_class, n_query, rounding_mode="floor")

        # # mean vs query
        # y_hat = idx_class.clone()
        
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

        return loss_val, acc_val


class TestResult:
    def __init__(self):
        self.acc_overall = torch.Tensor().to(_CG.DEVICE)
        self.y_hat = torch.Tensor().to(_CG.DEVICE)
        self.target_inds = torch.Tensor().to(_CG.DEVICE)

    def proto_test(self, recons, target, n_support):
        classes = torch.unique(target)
        n_classes = len(classes)
        mapping = {i: classes[i].item() for i in range(n_classes)}      # mapping is necessary only if n_classes < all_classes

        # assuming n_query, n_target constants
        numel_set = len(torch.nonzero(target == classes[0]).view(-1))   # numel for support + query
        n_query = numel_set - n_support

        # retrieve support and query indexes
        support_idxs, query_idxs = [], torch.LongTensor().to(_CG.DEVICE)
        for c in classes:
            s, q = torch.split(torch.nonzero(target == c).view(-1), [n_support, n_query])
            support_idxs.append(s)  # 3 tensors with 5 samples each
            query_idxs = torch.cat((query_idxs, q))

        # use retrieved indexes to compute mean of 5 (idx_list) elements per class (output.size = n_classes * flatten_features)
        prototypes = torch.stack([recons[idx_list].mean(0) for idx_list in support_idxs])
        query_samples = recons[query_idxs.view(-1)]
        dists = ProtoTools.euclidean_dist(query_samples, prototypes)   # dim: (n_cls * sam_per_class, n_classes)

        # softmax of negative distance otherwise the softmax is negative (the highest value must be the closest)
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
    

    def proto_test2(self, s_batch: torch.Tensor, q_batch: torch.Tensor):
        n_classes, n_query, n_feat = (q_batch.shape)
        mapping = {i: i for i in range(n_classes)}

        protos = torch.mean(s_batch, dim=1)

        proto_dists = ProtoTools.euclidean_dist(s_batch.view(-1, n_feat), protos)
        query_dists = ProtoTools.euclidean_dist(q_batch.view(-1, n_feat), protos)
        mean_dists = torch.mean(proto_dists.view(n_classes, -1, n_classes), dim=1)
        
        # mean vs query
        mean_exp = mean_dists.unsqueeze(0).expand(query_dists.size(0), -1, -1)
        query_exp = query_dists.unsqueeze(1).expand(-1, mean_dists.size(1), -1)
        dis = torch.sqrt(torch.pow(mean_exp - query_exp, 2).sum(2))

        log_p_y = F.log_softmax(-dis, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes).to(_CG.DEVICE)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        _, idx_class = log_p_y.max(2)

        # mean vs query
        y_hat = idx_class.clone()

        acc_overall = y_hat.eq(target_inds.squeeze(2)).float().mean()
        acc_vals = { c: y_hat[c].eq(target_inds.squeeze(2)[c]).float().mean() for c in range(n_classes) }

        self.acc_overall = torch.cat((self.acc_overall, acc_overall.flatten()))
        self.y_hat = torch.cat((self.y_hat, y_hat.flatten()))
        self.target_inds = torch.cat((self.target_inds, target_inds.flatten()))

        return acc_overall, { v: acc_vals[i] for i, v in enumerate(mapping.values()) }
