# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import random
from utils import add_noise
from config import opt


class Train_Data(data.Dataset):
    def __init__(self, data_root1):

        self.transform = T.ToTensor()
        self.transform1 = T.ToPILImage()
        self.data_root1 = data_root1
        # self.data_root2 = data_root2

    def __getitem__(self, index):
        img_index = random.randint(1, 400)
        if img_index < 10:
            img = Image.open(self.data_root1 + "/" + "test_00" + str(img_index) + '.png')
        elif img_index < 100:
            img = Image.open(self.data_root1 + "/" +  "test_0" + str(img_index) + '.png')
        elif img_index <10000:
            img = Image.open(self.data_root1 + "/" +  "test_" + str(img_index) + '.png')
        img_H = img.size[0]
        img_W = img.size[1]
        H_start = random.randint(0, img_H - opt.crop_size)
        W_start = random.randint(0, img_W - opt.crop_size)
        crop_box = (W_start, H_start, W_start + opt.crop_size, H_start + opt.crop_size)
        img_crop = img.crop(crop_box)

        # n_crop = n.crop(crop_box)

        label = self.transform(img_crop)
        # noise = add_noise(label, opt.noise_level)
        noise = add_noise(label)
        # noise = self.transform(noise)
        return noise, label

    def __len__(self):
        return opt.num_data


if __name__ == '__main__':
    train_data = Train_Data(data_root1=opt.data_root1,data_root2=opt.data_root2)

    train_loader = DataLoader(train_data, 1)

    for i, (data, label) in enumerate(train_data):
        print(i)
        if i == 0:
            print(data)
            print(label)
            print(data.size())
            print(label.size())
            break





