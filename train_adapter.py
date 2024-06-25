import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import torch
import sys
import datetime
import config
import random
import numpy as np
import torchvision
import torch.distributed as dist
from time import perf_counter
from pathlib import Path
from tqdm import tqdm
from network_adapter import network
from logger import get_logger
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import Vimeo
current_path = Path(__file__).resolve().parents[0]
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generative model training for VQ-VAE")
    parser.add_argument("--DDP", default=False, action="store_true", help="if using data distributed parallel")
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int, help="used to identify main process and subprocess")
    parser.add_argument("--tag", default=datetime.datetime.now())
    parser.add_argument("--config_path", default=current_path / "config.yaml")
    parser.add_argument("--save_path", default=current_path / "outputs")
    parser.add_argument("--data_path", default=current_path / "datasets")
    parser.add_argument("--load_path", default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = get_logger("train", args.save_path / f"train_{args.tag}.log")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if config.seed != None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # if use multi-GPU

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)

    net = network(config).to(device)

    # ------------------------------load weight---------------------------
    weight = torch.load("./outputs/VQVAE_2024-03-31 12:02:40.345042.pt", map_location="cpu")
    net.load_state_dict(weight, strict=True)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    net = net.module

    optim = torch.optim.Adam(net.parameters(), lr=float(config["lr"]), amsgrad=True)
    data_transform = Compose([torchvision.transforms.RandomCrop(256, pad_if_needed=True), ToTensor()])
    train_dataset = Vimeo(Path(args.data_path))
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=False, pin_memory=False, sampler=sampler)

    net.train()
    for epoch in range(config.epochs):
        if dist.get_rank() == 0:
            train_bar = tqdm(train_loader)
        else:
            train_bar = train_loader

        running_loss, running_psnr, running_perplexity = [0] * 3
        num = 0

        for data in train_bar:
            num += 1
            optim.zero_grad()

            data = data.cuda()
            pack = net(data)
            x_hat = pack["x_hat"]

            recon_loss = torch.mean((x_hat - data) ** 2) / 1
            loss = recon_loss + pack["kl_loss"]

            loss.backward()
            optim.step()

            mse = torch.mean((data - x_hat) ** 2)
            psnr = torch.mean(10 * torch.log10(1. / mse)) 
            running_loss += loss.cpu().detach().numpy()
            running_psnr += psnr.cpu().detach().numpy()

            if dist.get_rank() == 0:
                if num % config["log_interval"] == 0:
                    logger.info(f"loss: {running_loss / num}\n psnr: {running_psnr / num}\n")

                train_bar.desc = f"{epoch} | {config['epochs']}"

        ckpt = net.state_dict()
        torch.save(ckpt, args.save_path / f"baseline_{args.tag}.pt")
        save_image(x_hat, args.save_path / "vision" / f"{epoch}.png")


            
