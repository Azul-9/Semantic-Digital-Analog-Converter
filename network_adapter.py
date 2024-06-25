import torch.nn as nn
from subnets import *
from adapter import Adapter
import config as cf
from bsc_channel import BSC_channel


class network(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_channel = config.hidden_channel  # 128
        res_hidden_channel = config.res_hidden_channel  # 32
        n_res_layers = config.n_res_layers  # 2

        self.encoder = Encoder(3, hidden_channel, n_res_layers, res_hidden_channel)
        
        self.adapter = Adapter(cf)
        self.channel = BSC_channel(stochastic=False, bit_flip_prob=3.9e-6)

        self.decoder = Decoder(hidden_channel, hidden_channel, n_res_layers, res_hidden_channel)

    def forward(self, x):
        z_e = self.encoder(x)

        pack = self.adapter.encode(z_e)
        bit_index = pack["out"]
        x_reshape = pack["x"]
        pack = self.channel(bit_index)
        bit_index_noise = pack["out"]
        ber = pack["bit_flip_prob"]
        out_feature = self.adapter.decode(bit_index_noise)
        out_feature = torch.reshape(out_feature, x_reshape.shape)

        kl_loss = torch.mean((out_feature.detach() - x_reshape) ** 2) + 0.25 * torch.mean((out_feature - x_reshape.detach()) ** 2)
        bit_out = x_reshape + (out_feature - x_reshape).detach()
        bit_out = torch.permute(bit_out, (0, 3, 1, 2))

        recon_out = self.adapter.after_quant(bit_out)

        x_hat = self.decoder(recon_out)

        return dict(
            kl_loss=kl_loss,
            x_hat=x_hat,
            ber=ber,
            z_e=z_e,
            recon_out=recon_out
        )