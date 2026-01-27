# -----------------------------------------------------------------------------------
# 파일명       : Origin_UNet.py
# 설명         : Image Segmentation(U-Net)을 통한 소아 복부 질환 탐지       
# 작성자       : 이민하
# 작성일       : 2025-08-26
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 정상 1, 비정상 4가지의 상황을 Image Segmentation을 통해 구별
# - Computer Vision 분야의 U-Net 구조를 직접 커스텀하여 구현 
#
# >> 성능
# Train Multi-Class Dice Score : 0.898 -> 0.908
# Test Multi-Class Dice Score : 0.883 -> 0.912
# -----------------------------------------------------------------------------------


import os
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader

import albumentations as A

from functools import reduce

from XRaySegModules import *

# Config 파일 불러오기
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")

config = load_config(CONFIG_PATH)

# 데이터 경로 설정
TRAIN_DATA_PATH = config["path"]["train"]["source"]
TRAIN_LABEL_PATH = config["path"]["train"]["label"]

VAL_DATA_PATH = config["path"]["val"]["source"]
VAL_LABEL_PATH = config["path"]["val"]["label"]

TEST_DATA_PATH = config["path"]["test"]["source"]
TEST_LABEL_PATH = config["path"]["test"]["label"]

# Training 데이터 준비
folder_list = []
label_file_list = []
label_list = []

for folder in os.listdir(TRAIN_LABEL_PATH):
    folder_list.append(os.path.join(TRAIN_LABEL_PATH, folder))

for path in folder_list:
    