from typing import Tuple
import torch
import torch.nn as nn
import kornia.augmentation as K

class RandomTransform(nn.Module):
    #size is the desired output size (out_h, out_w) of the crop.
    def __init__(self, size: Tuple[int, int], randcrop: bool, augment: bool):
        super().__init__()
        if randcrop:
            self.crop = K.RandomCrop(size) #Crop random patches of a tensor image on a given size.
            #output size (Batch, Color, out_h, out_w)
        else:
            self.crop = K.CenterCrop(size)
        #random flip with probability of 0.5
        self.flip = nn.Sequential(K.RandomVerticalFlip(p=0.5),
                                  K.RandomHorizontalFlip(p=0.5))
        self.augment = augment

    def forward(self, img, disparity, conf=None):
        if conf is None:
            input = torch.cat([img, disparity], dim=0)
        else:
            input = torch.cat([img, disparity, conf], dim=0)
        #randomly crop and flip
        input = self.crop(input)
        if self.augment:
            input = self.flip(input)
        img = input[:, :3]
        disparity = input[:, [3]]
        if conf is None:
            return img, disparity
        else:
            conf = input[:, [4]]
            return img, disparity, conf