import torch
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class Vimeo(Dataset):
    def __init__(self, data_path:Path, if_train=True, full_mode=True):
        if not full_mode:
            assert os.path.exists(data_path / "data_list.txt"), f"data_list.txt is missing or {data_path} is error"
            data_list = "data_list.txt"
        else:
            assert os.path.exists(data_path / "data_list_full.txt"), f"data_list_full.txt is missing or {data_path} is error"
            data_list = "data_list_full.txt"
        super(Vimeo, self).__init__()
        self.if_train = if_train
        self.data_path = str(data_path.absolute())

        with open(data_path / data_list, "r") as f:
            self.input_data = f.readlines()

        print(f"ready to load {len(self.input_data)} sequences for training")

    def __len__(self):
        return len(self.input_data)
    
    def get_transform(self, train_flag):
        if train_flag:
            transform = T.Compose([
                T.RandomCrop(256, pad_if_needed=True),
                T.ToTensor()
            ])
        else:
            transform = T.Compose([T.ToTensor()])
        return transform

    def __getitem__(self, index):
        transform = self.get_transform(self.if_train)
        img_path = self.data_path + "/sequences/" + self.input_data[index].rstrip()
        img = Image.open(img_path).convert("RGB")

        img = transform(img)
        out_image = img * 2 - 1  # normalize to [-1, 1]
        
        return out_image