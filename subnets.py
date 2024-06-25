import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualLayer(nn.Module):
    def __init__(self, in_channel, hidden_channel, res_hidden_channel) -> None:
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, res_hidden_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(res_hidden_channel, hidden_channel, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out = x + self.res_block(x)
        return out


class ResidualStack(nn.Module):
    def __init__(self, in_channel, hidden_channel, res_hidden_channel, n_res_layers) -> None:
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_channel, hidden_channel, res_hidden_channel)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_res_layers, res_hidden_channel) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel // 2, hidden_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channel, hidden_channel, res_hidden_channel, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_res_layers, res_hidden_channel) -> None:
        super().__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channel, hidden_channel, res_hidden_channel, n_res_layers),
            nn.ConvTranspose2d(hidden_channel, hidden_channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channel // 2, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    

    
