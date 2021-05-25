import torch
import torch.nn as nn
import torch.nn.functional as F


class THLSum(nn.Module):
    def _apply_l2_norm(self, x):
        # Calculate L2 norm of a vector and apply normalization
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return torch.div(x, norm)

    def forward(self, x):
        x_norm = self._apply_l2_norm(x)
        dist = F.pdist(x_norm, p=2)
        loss = torch.sum(1 / dist)
        return loss


class THLMaxCosine(nn.Module):
    def forward(self, x):
        cos_mat = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        cos_mat -= 2 * torch.diag(torch.diag(cos_mat))
        loss = torch.max(cos_mat, dim=1)[0]
        return loss.mean()


class THLMaxInverseDist(nn.Module):
    def fast_cdist(self, x1, x2):
        # https://github.com/pytorch/pytorch/pull/25799
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        # x1 and x2 should be identical in all dims except -2 at this point
        # Compute squared distance matrix using quadratic expansion
        # But be clever and do it with a single matmul call
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))
        # Zero out negative values
        res.clamp_min_(1e-30).sqrt_()
        return res

    def _apply_l2_norm(self, x):
        # Calculate L2 norm of a vector and apply normalization
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return torch.div(x, norm)

    def forward(self, x):
        x_norm = self._apply_l2_norm(x)
        dist_mat = self.fast_cdist(x_norm, x_norm)
        dist_mat += 10 * torch.diag(1-torch.diag(dist_mat))
        loss = torch.max(1 / dist_mat, dim=1)[0]
        return loss.mean()
