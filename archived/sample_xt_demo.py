#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


# model parameters
alpha = 0.99

# structural constants
xdim = 100


def sample_xt(zt):
    xt_mode = (zt + xdim / 2).clamp(0, xdim - 1).floor().long()
    xt_params = F.one_hot(xt_mode, num_classes=xdim) * alpha + (1 - alpha) / xdim
    xt_rv = Categorical(xt_params)
    xt_sample = xt_rv.sample()
    xt_ll = xt_rv.log_prob(xt_sample)
    return xt_sample, xt_ll

if __name__ == "__main__":
    # sample z_t ~ Normal(0, 10)
    zt = torch.randn(2000) * 10

    # sample x_t | z_t
    xt_sample, xt_ll = sample_xt(zt)

    # report the result
    print("Here is a sample of x_t: ", xt_sample)
