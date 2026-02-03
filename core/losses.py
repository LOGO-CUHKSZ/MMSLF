import torch
from torch import nn


class MultimodalLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, out_student, out_teacher, label):
        loss_reg = torch.nn.L1Loss()(out_student['preds'], label)
        loss_attn = torch.nn.L1Loss()(out_student['h_la_attn'][-1], out_teacher['h_la_attn'][-1]) + \
              torch.nn.L1Loss()(out_student['h_lv_attn'][-1], out_teacher['h_lv_attn'][-1])
        loss_fusion = torch.nn.L1Loss()(out_student['h_reg'][:, :4], out_teacher['h_reg'][:, :4])

        loss = self.alpha * loss_reg + self.beta * loss_attn + self.gamma * loss_fusion

        return {'loss': loss, 'loss_reg': loss_reg, 'loss_attn': loss_attn, 'loss_fusion': loss_fusion}
