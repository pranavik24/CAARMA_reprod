#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
from .utils import accuracy
from helper.diffusion_mixup import diffusion_mixup
from helper.mixup_avg import mixup_data_euc_avg

class amsoftmax_gan(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        margin=0.2,
        scale=30,
        synthetic_strategy="avg",
        diffusion_timesteps=100,
        diffusion_t_min=1,
        diffusion_t_max=20,
        diffusion_beta_start=0.0001,
        diffusion_beta_end=0.02,
        diffusion_embedding_noise=0.0,
        diffusion_fake_fraction=1.0,
        **kwargs,
    ):
        super(amsoftmax_gan, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.synthetic_strategy = str(synthetic_strategy).strip().lower()
        self.diffusion_timesteps = diffusion_timesteps
        self.diffusion_t_min = diffusion_t_min
        self.diffusion_t_max = diffusion_t_max
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_embedding_noise = diffusion_embedding_noise
        self.diffusion_fake_fraction = diffusion_fake_fraction
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))
        if self.synthetic_strategy == "diffusion":
            print(
                "Diffusion mixup enabled: "
                f"timesteps={self.diffusion_timesteps}, "
                f"t=[{self.diffusion_t_min}, {self.diffusion_t_max}]"
            )

    def forward(self, x, label=None, flagSyn=False):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        if self.synthetic_strategy == "diffusion":
            synthetic_embeddings, y_combined, w_combined = diffusion_mixup(
                x,
                self.W,
                label,
                diffusion_timesteps=self.diffusion_timesteps,
                diffusion_t_min=self.diffusion_t_min,
                diffusion_t_max=self.diffusion_t_max,
                diffusion_beta_start=self.diffusion_beta_start,
                diffusion_beta_end=self.diffusion_beta_end,
                diffusion_embedding_noise=self.diffusion_embedding_noise,
                diffusion_fake_fraction=self.diffusion_fake_fraction,
            )
        else:
            synthetic_embeddings,  y_combined , w_combined = mixup_data_euc_avg(
                x, self.W, label
                )
        if flagSyn:
            
            x_combined_0 = synthetic_embeddings.to(x.device) #torch.cat((x, synthetic_embeddings), dim=0)
            w_combined_0 = w_combined.to(x.device) #torch.cat((self.W.to(x.device), w_combined.to(x.device)), dim=1)
            y_combined_0 = y_combined.to(x.device) #torch.cat((label.to(x.device), y_combined.to(x.device)), dim=0)

            x_norm = torch.norm(x_combined_0, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x_combined_0, x_norm)
            w_norm = torch.norm(w_combined_0, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(w_combined_0, w_norm)
            costh = torch.mm(x_norm, w_norm)
            label_view = y_combined_0.view(-1,1) #label.view(-1, 1)
            if label_view.is_cuda: label_view = label_view.cpu()
            delt_costh = torch.zeros(
                costh.size(), device=costh.device, dtype=costh.dtype
            ).scatter_(1, label_view.to(costh.device), self.m)
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            
            loss = self.ce(costh_m_s, y_combined_0) #label)
            
            final_loss = (loss)
            acc = accuracy(costh_m_s.detach(), y_combined_0.detach(), topk=(1,))[0]
            return final_loss, acc, synthetic_embeddings
        else: 

            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)
            costh = torch.mm(x_norm, w_norm)
            label_view =  label.view(-1, 1)
            if label_view.is_cuda: label_view = label_view.cpu()
            delt_costh = torch.zeros(
                costh.size(), device=costh.device, dtype=costh.dtype
            ).scatter_(1, label_view.to(costh.device), self.m)
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            
            loss = self.ce(costh_m_s,  label)
            
            final_loss = (loss)
            acc = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
            return final_loss, acc, synthetic_embeddings
