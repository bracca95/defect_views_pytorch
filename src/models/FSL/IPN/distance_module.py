import torch
import torch.nn.functional as F

from torch import nn


class DistScale(nn.Module):
    
    def __init__(self, in_len: int, k_shot: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_len, k_shot),
            nn.Sigmoid()
        )

    @staticmethod
    def cat_support_query(support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        """cat support vector with query vector

        The concatenation is performed according to the Improved Protonet paper (SeeAlso section). Every query is
        concatenated with each of the k samples of every class. This means that the support set must be replicated
        first according to the number of queries.

        Args:
            support_query (torch.Tensor): the support set, size = (batch, n_support, feature_vector)
            query_set (torch.Tensor): the query set, size = (batch, n_queries, feature_vector)

        Returns:
            torch.Tensor

        SeeAlso:
            [Improved Protonet paper](https://www.sciencedirect.com/science/article/pii/S0167865520302610)
        """
        
        n_supports = support_set.shape[1]
        n_queries = query_set.shape[1]
        
        # replicate the whole support for every query. Replicate every query for all the k-th support sample
        support_expand = support_set.repeat(1, n_queries, 1)
        query_expand = torch.repeat_interleave(query_set, repeats=n_supports, dim=1)

        return torch.cat((support_expand, query_expand), dim=-1)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, dim=1)
        #return F.softmax(out, dim=0)
        return out