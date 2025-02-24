#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:14:29 2020
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""
from torch.utils.data.dataset import Dataset
import skimage.io
#from skimage.metrics import normalized_mutual_information
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import cv2
import os
from argument import Transform
from spectral import *
from spectral import open_image
import random
import math
from scipy.ndimage import zoom
import warnings
import tiff
warnings.filterwarnings('ignore')
from einops import repeat

class Data_Generate_Cho(Dataset):#
    def __init__(self, img_paths, seg_paths=None,
                 cutting=None, transform=None,
                 channels=None, outtype='3d', envi_type='img',
                 multi_class= 1):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.transform = transform
        self.cutting = cutting
        self.channels = channels
        self.outtype = outtype
        self.envi_type = envi_type
        self.multi_class = multi_class

    def __getitem__(self,index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]
        
        mask = np.load(mask_path)[:32, :32]

        img = np.load(img_path)[:32, :32, :]

        mask = mask.astype(np.uint8)
        # if self.cutting is not None:
        #     xx = random.randint(0, img.shape[0] - self.cutting)
        #     yy = random.randint(0, img.shape[1] - self.cutting)
        #     patch_img = img[xx:xx + self.cutting, yy:yy + self.cutting]
        #     patch_mask = mask[xx:xx + self.cutting, yy:yy + self.cutting]
        #     img = patch_img
        #     mask = patch_mask


        img = img[:, :, None] if len(img.shape)==2 else img

        img = np.transpose(img, (2, 0, 1))

        if self.outtype == '3d':
            img = img[None]

        # mask = mask[None, ]

        mask = mask[None, ].astype(np.float32)
        img = img.astype(np.float32)

        return img, mask
            
    def __len__(self):
        return len(self.img_paths)