import torch
import os
import config
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # set before import other custom module, otherwise it would be useless
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
from network_adapter import network
from tqdm import tqdm



dataset = ImageFolder("./datasets/", transform=Compose([ToTensor()]))
dataloader = DataLoader(dataset, 1, shuffle=False)
dataloader = tqdm(dataloader)
net_weight = torch.load("./outputs/history/baseline_1bit.pt", map_location="cpu")
net = network(config).cuda()
net.load_state_dict(net_weight, strict=True)
net.eval()
psnr_avg = 0
kl_avg = 0
end_avg = 0


for index, data in enumerate(dataloader):
    data = data[0].cuda()
    out = net(data)
    x_hat = out["x_hat"]
    kl_loss = out["kl_loss"]
    z_e = out["z_e"]
    recon_out = out["recon_out"]
    mse = torch.mean((data - x_hat) ** 2)
    mse_end = torch.mean((z_e - recon_out) ** 2)
    psnr = torch.mean(10 * torch.log10(1. / mse)) 
    psnr_avg += psnr.detach()  # it's important for correct gpu memory free, i mean detach()
    kl_avg += torch.sqrt(kl_loss / 1.25).detach()
    end_avg += torch.sqrt(mse_end).detach()

print(psnr_avg / index + 1)
print(kl_avg / index + 1)
print(end_avg / index + 1)
