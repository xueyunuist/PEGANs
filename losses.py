import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = self.bce(pred_real, torch.ones_like(pred_real))
            loss_fake = self.bce(pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss1 = self.bce(pred_real, torch.zeros_like(pred_real))
            loss = -1 * loss1
            return loss


class BCEWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = self.bce(pred_real, torch.ones_like(pred_real))
            loss_fake = self.bce(pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = self.bce(pred_real, torch.ones_like(pred_real))
            return loss


class HingeLoss(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class LeastSquareLoss(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_real))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = F.mse_loss(pred_real, torch.ones_like(pred_real))
            return loss


class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = -loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = self.bce(
                (pred_real + 1) / 2, torch.ones_like(pred_real))
            loss_fake = self.bce(
                (pred_fake + 1) / 2, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = self.bce(
                (pred_real + 1) / 2, torch.ones_like(pred_real))
            return loss
