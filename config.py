# coding = utf-8
# -*- coding:utf-8 -*-
import random
import numpy as np
import torch
import os

root_path = os.getcwd().replace('\\', '/') + '/'
train_path = root_path + 'data/train.txt'
test_without_label_path = root_path + 'data/test_without_label.txt'
data_path = root_path + 'data/data/'
train_combined_path = root_path + 'data/input/train_data.json'
test_combined_path = root_path + 'data/input/test_data.json'
pretrained_model_img = 'google/vit-base-patch16-224-in21k'
pretrained_model_text = 'bert-base-multilingual-cased'
cache_model_path = root_path + 'cache/model'
prediction_path = root_path + 'cache/prediction.txt'

batch_size = 32
epoch = 5
lr = 1e-3
img_lr = 2e-5
text_lr = 2e-5
max_len = 128


def setup_seed():
    torch.manual_seed(1511)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1511)
    np.random.seed(1511)
    random.seed(1511)
