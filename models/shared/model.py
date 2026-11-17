import torch
from torch import nn
from patient_model import AttentionPool

class Model(nn.Module):

    def __init__(self, cd, clinical, image):
        super().__init__()
        self.c_mode = clinical
        self.i_mode = image
        self.visual = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.patch = AttentionPool(64)
        self.image = AttentionPool(64)
        self.clinical = nn.Sequential(nn.Linear((cd - 1 if clinical in ['knowledge', 'no_neighbor', 'random_gat', 'both'] else cd) or 1, 64), nn.ReLU())
        self.cv = nn.Linear(512, 64)
        self.cs = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, mask, c, e, g, detail=False):
        n = len(c)
        zero = c.new_zeros((n, 64))
        pa = None
        ia = None
        ca = None
        if self.i_mode != 'none':
            v = x.new_zeros((*mask.shape, 64))
            v[mask] = self.visual(x)
            (im, pa) = self.patch(v, mask)
            (vi, ia) = self.image(im, mask.any(-1))
        else:
            vi = zero
        if self.c_mode in ['knowledge', 'no_neighbor', 'random_gat']:
            cl = self.clinical(c[:, :-1]) * c[:, -1:]
        elif self.c_mode == 'both':
            cl = self.clinical(torch.cat([c[:, :-513], c[:, -513:-1] * c[:, -1:]], -1))
        else:
            cl = self.clinical(c) if self.c_mode != 'none' else zero
        if self.i_mode in ['name', 'concat']:
            cv = self.cv(e)
            ca = torch.softmax(self.cs(cv).squeeze(-1), -1) * g
            ca = ca / ca.sum(-1, keepdim=True).clamp_min(1e-12)
            ki = (ca.unsqueeze(-1) * cv).sum(1)
        else:
            ki = zero
        z = torch.cat([cl, vi, ki], -1)
        logit = self.head(z).squeeze(-1)
        return dict(logit=logit, embedding=z, clinical=cl, image=vi, concept=ki, patch_attention=pa, image_attention=ia, concept_attention=ca) if detail else logit
