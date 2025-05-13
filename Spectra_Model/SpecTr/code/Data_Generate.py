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

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]
        
        # Load data
        mask = np.load(mask_path)[:32, :32]
        img = np.load(img_path)[:32, :32, :]  # Shape: [32, 32, 136]
        
        # Clean mask
        mask[mask == 190] = 0  # your old ignore-mask
        # new: keep only 1–5, zero everything else
        valid = np.isin(mask, [1, 2, 3, 4, 5])
        mask[~valid] = 0  # set all non-(1..5) to 0
        
        # Convert mask to proper type
        mask = mask.astype(np.int64)  # Good for CrossEntropy
        
        # MISSING: Process image dimensions
        # Current img shape: [32, 32, 136] (height, width, spectral)
        # Need: [1, 136, 32, 32] (channels, spectral, height, width)
        
        # Step 1: Transpose to [spectral, height, width]
        img = np.transpose(img, (2, 0, 1))  # Now: [136, 32, 32]
        
        # Step 2: Add channel dimension
        img = img[None, ...]  # Now: [1, 136, 32, 32]
        
        # Convert to float32
        img = img.astype(np.float32)
        
        return img, mask
            # img = img[:, :, None] if len(img.shape)==2 else img

            # img = np.transpose(img, (2, 0, 1))

            # if self.outtype == '3d':
            #     img = img[None]

            # # mask = mask[None, ]

            # mask = mask[None, ].astype(np.float32)
            # img = img.astype(np.float32)

            # return img, mask
            
    def __len__(self):
        return len(self.img_paths)