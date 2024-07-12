import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.covariance import EmpiricalCovariance
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
from functools import partial, reduce
from typing import Callable, List, Literal
_logger = logging.getLogger(__name__)



def sync_tensor_across_gpus(t: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all gpus and return a tensor with dim 0 equal to the number of gpus.

    Args:
        t (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_

    References:
        https://discuss.pytorch.org/t/ddp-evaluation-gather-output-loss-and-stuff-how-to/130593/2
    """
    if not dist.is_initialized():
        return t
    group = dist.group.WORLD
    group_size = dist.get_world_size(group)
    if group_size == 1:
        return t
    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]
    dist.all_gather(gather_t_tensor, t)
    gather_t_tensor = torch.cat(gather_t_tensor, dim=0)

    return gather_t_tensor


def get_composed_attr(model, attrs: List[str]):
    return reduce(lambda x, y: getattr(x, y), attrs, model)


class ViM:
    """Virtual Logit Matching (ViM) detector.

    Args:
        model (torch.nn.Module): Model to be used to extract features

    References:
        [1] https://arxiv.org/abs/2203.10807
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs
    ):
        self.model = model
        self.model.eval()

        # create feature extractor
        self.feature_extractor = self.model.pen_ultimate_layer

        # get the model weights of the last layer
        last_layer = model.fc
        self.w = model.fc.weight.data.squeeze().clone()
        self.b = model.fc.bias.data.squeeze().clone()

        _logger.debug("w shape: %s", self.w.shape)
        _logger.debug("b shape: %s", self.b.shape)

        self.head = list(model._modules.values())[-1]

        # new origin
        self.u = -torch.matmul(torch.linalg.pinv(self.w), self.b).float()
        _logger.debug("New origin shape: %s", self.u.shape)

        self.principal_subspace = None
        self.train_features = []
        self.train_logits = []
        self.alpha = None
        self.top_k = None

    def _get_logits(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits

    def start(self, *args, **kwargs):
        self.principal_subspace = None
        self.train_features = None
        self.train_logits = None
        self.alpha = None
        self.top_k = None

    @torch.no_grad()
    def update(self, x: torch.Tensor,  *args, **kwargs):
        features = self.feature_extractor(x)
        features = sync_tensor_across_gpus(features)
        if dist.is_initialized():
            dist.gather(features, dst=0)

        if self.train_features is None:
            self.train_features = torch.flatten(features, start_dim=1).cpu()
        else:
            self.train_features = torch.cat([self.train_features, torch.flatten(features, start_dim=1).cpu()], dim=0)

        if self.train_logits is None:
            self.train_logits = self._get_logits(x).cpu()
        else:
            self.train_logits = torch.cat([self.train_logits, self._get_logits(x).cpu()])

    def end(self):
        self.top_k = 1000 if self.train_features.shape[1] > 1500 else 512

        _logger.info("Train features shape: %s", self.train_features.shape)
        _logger.info("Train logits shape: %s", self.train_logits.shape)

        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.train_features.cpu().numpy() - self.u.detach().cpu().numpy())
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        determinant = np.linalg.det(ec.covariance_)
        _logger.debug("Determinant: %s", determinant)
        _logger.debug("Eigen values: %s", eig_vals)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[ :]
        self.principal_subspace = torch.from_numpy(
            np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)
        ).float()
        _logger.debug("Principal subspace: %s", self.principal_subspace)

        # calculate residual
        x_p_t = torch.matmul(self.train_features.cpu() - self.u.cpu(), self.principal_subspace.cpu())
        vlogits = torch.norm(x_p_t, dim=-1)
        self.alpha = self.train_logits.max(dim=-1)[0].mean() / vlogits.mean()
        _logger.debug("Alpha: %s", self.alpha)

        del self.train_features

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)

        logits = self._get_logits(x)

        x_p_t = torch.norm(
            torch.matmul(torch.flatten(features, 1) - self.u.to(x.device), self.principal_subspace.to(x.device)), dim=-1
        )
        vlogit = x_p_t * self.alpha
        energy = torch.logsumexp(logits, dim=-1)
        score = -vlogit + energy
        return torch.nan_to_num(score, 1e6)