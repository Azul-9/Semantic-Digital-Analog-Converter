"""
it's a semantic digital-analog-converter , used to convert semantic tensor and digital bits bidirectionally
"""


import torch
import torch.nn as nn

    
def mapper_generator(quant_num_bits):  # used to generate a mapping matrix to map the index to the corresponding code
    temp = torch.empty(2 ** quant_num_bits, quant_num_bits)

    for i in range(2 ** quant_num_bits):
        binary_str = bin(i)[2: ]
        if len(binary_str) < quant_num_bits:
            binary_str = "0" * (quant_num_bits - len(binary_str)) + binary_str
        binary_tensor = torch.tensor([int(x) for x in binary_str])
        temp[i, : len(binary_tensor)] = binary_tensor

    return temp

def binary_rows_to_decimal(binary_tensor):  # Convert binary rows to decimal values
    num_bits = binary_tensor.shape[1]
    powers_of_two = 2 ** torch.flip(torch.arange(num_bits, dtype=binary_tensor.dtype, device=binary_tensor.device), dims=(0,))
    decimal_values = torch.sum(binary_tensor * powers_of_two, dim=1)

    return decimal_values.long()


class Adapter(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_adapter = config.use_adapter
        self.quant_num_bits = config.quant_num_bits
        self.mapper = nn.Parameter(mapper_generator(self.quant_num_bits), requires_grad=False)
        self.code_book = nn.Embedding(2 ** self.quant_num_bits, self.quant_num_bits, device="cuda")
        self.code_book.weight.data.uniform_(-1.0 / 2 ** self.quant_num_bits, 1.0 / 2 ** self.quant_num_bits)
        self.pre_quant = nn.Conv2d(config.hidden_channel, self.quant_num_bits * config.hidden_channel, kernel_size=1, groups=config.hidden_channel)
        self.after_quant = nn.Conv2d(self.quant_num_bits * config.hidden_channel, config.hidden_channel, kernel_size=1, groups=config.hidden_channel)

    def forward(self, x):
        # the input of this network should be a 4-D tensor with shape like (batch_size, channel, height, width)
        x = self.pre_quant(x)  # transform shape from [batch_size, channel, height, width] to [batch_size, quant_num_bits * channel, height, width]

        x = torch.permute(x, (0, 2, 3, 1))  # shape [batch_size, height, width, quant_num_bits * channel]
        x_flatten = torch.reshape(x, (-1, self.quant_num_bits))  # shape [batch_size * height * width * quant_num_bits * channel, quant_num_bits]

        # distances from x_flatten to code book e: (x - e)^2 = x^2 + e^2 - 2 e * x
        euclid_distance = torch.sum(x_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.code_book.weight ** 2, dim=1) - 2 * torch.matmul(x_flatten, self.code_book.weight.t())

        min_index = torch.argmin(euclid_distance, dim=1).unsqueeze(1)
        temp = torch.zeros(min_index.shape[0], 2 ** self.quant_num_bits).cuda().scatter(1, min_index, 1)  # create a one-hot matrix to select corresponding code

        out = torch.matmul(temp, self.code_book.weight)
        out = torch.reshape(out, x.shape)

        kl_loss = torch.mean((out.detach() - x) ** 2) + 0.25 * torch.mean((out - x.detach()) ** 2)  # try to force encoder's output to be close to the code_book
        bit_out = x + (out - x).detach()  # straight-through gradient
        bit_out = torch.permute(bit_out, (0, 3, 1, 2))
        recon_out = self.after_quant(bit_out)

        return dict(
            kl_loss=kl_loss,
            bit_out=bit_out,
            recon_out=recon_out
        )
    
    def encode(self, x):  # here x is the output of the encoder
        x = self.pre_quant(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x_flatten = torch.reshape(x, (-1, self.quant_num_bits))

        euclid_distance = torch.sum(x_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.code_book.weight ** 2, dim=1) - 2 * torch.matmul(x_flatten, self.code_book.weight.t())

        min_index = torch.argmin(euclid_distance, dim=1)
        out = self.mapper[min_index]

        return dict(
            out=out,
            x=x
        )
    
    def decode(self, x):  # here x is the two-value bit matrix
        index = binary_rows_to_decimal(x).unsqueeze(1)
        temp = torch.zeros(index.shape[0], 2 ** self.quant_num_bits).cuda().scatter(1, index, 1)  # create a one-hot matrix to select corresponding code

        out = torch.matmul(temp, self.code_book.weight)

        return out

