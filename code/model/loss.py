import torch
from torch import nn
import utils.general as utils


class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
        }

        return output
