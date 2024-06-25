import torch
import random
import torch.nn as nn


class BSC_channel(nn.Module):
    def __init__(self, stochastic=False, bit_flip_prob=0, max_ber=0.3) -> None:
        super().__init__()
        if stochastic:
            assert bit_flip_prob == 0, "conflict argument between stochastic and bit_flip_prob"
        self.stochastic = stochastic
        self.bit_flip_prob = bit_flip_prob
        self.max_ber = max_ber
        self.flag = True

    def forward(self, x):  # this func is compatible with {0, 1} bits, caution about the input format
        if self.flag:
            assert x.max() == 1 and x.min() == 0, "this model is only compatible with {0, 1} bits, please check the input format."
            self.flag = False

        if self.stochastic:
            self.bit_flip_prob = random.uniform(0, self.max_ber)  # change bit flip probability for each batch
        
        out = x.clone()
        noise = torch.rand_like(x) < self.bit_flip_prob
        out[noise] = 1 - out[noise]

        return dict(
            out=out,
            bit_flip_prob=self.bit_flip_prob
        )
        
