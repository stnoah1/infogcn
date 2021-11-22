import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import linalg as LA

def sym_reg(pred):
    loss = torch.zeros(pred.shape[0]).to(pred.device)
    l_bones = [(0, 12), (12, 13), (13, 14), (14, 15), (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21)]
    r_bones = [(0, 16), (16, 17), (17, 18), (18, 19), (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23)]
    for l_bone, r_bone in zip(l_bones, r_bones):
        l_bone_len = LA.norm(pred[:, l_bone[0], :] - pred[:, l_bone[1], :], 2, dim=-1)
        r_bone_len = LA.norm(pred[:, r_bone[0], :] - pred[:, r_bone[1], :], 2, dim=-1)
        loss += torch.abs(l_bone_len - r_bone_len)
    return torch.mean(loss)

def constraint_reg(pred, ratio=3):
    loss = torch.zeros(pred.shape[0]).to(pred.device)
    arms = [(9, 10), (5, 6), (17, 18), (13, 14)]
    hands = [(10, 11), (6, 7), (18, 19), (14, 15)]
    for arm, hand in zip(arms, hands):
        arm_len = LA.norm(pred[:, arm[0], :] - pred[:, arm[1], :], 2, dim=-1)
        hand_len = LA.norm(pred[:, hand[0], :] - pred[:, hand[1], :], 2, dim=-1)
        loss += torch.abs(arm_len/ratio - hand_len)
    return torch.mean(loss)


class ReconLoss(nn.Module):
    def __init__(self, p=2):
        super(ReconLoss, self).__init__()
        self.loss = nn.PairwiseDistance(p)

    def forward(self, pred, gt):
        B, V, C = pred.shape
        loss = self.loss(pred.contiguous().view(-1, C), gt.contiguous().view(-1, C))
        return loss.view(B, V).mean(-1).mean(-1)


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.loss = nn.CosineSimilarity(dim=2)

    def forward(self, pred, gt):
        return  self.loss(pred, gt).mean(1).mean(0)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]

