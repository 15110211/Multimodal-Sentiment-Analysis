# coding = utf-8
# -*- coding:utf-8 -*-
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTFeatureExtractor
import config

config.setup_seed()


tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


class MultiDataset(Dataset):
    def __init__(self, data: list, tokenizer: BertTokenizer, extractor: ViTFeatureExtractor, maxLen: int):
        self.data = data
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.maxLen = maxLen
    def __len__(self):
        return len(self.data)
    
    # 完成了对数据集样本的文本和图像的编码和特征提取过程
    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        text = self.data[item]['text']
        img = self.data[item]['img']
        tag = self.data[item]['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)  # 转换为Tensor并指定数据类型为torch.long
        encoding = self.tokenizer.encode_plus( # 对文本进行编码处理
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.maxLen,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        img = self.extractor(images=Image.open(config.data_path + img),return_tensors='pt') # 对图像进行特征提取处理
        return {
            'guid': guid,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'img': img,
            'tag': tag
        }


def getMultiDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_text)
    extractor = ViTFeatureExtractor.from_pretrained(config.pretrained_model_img)
    return MultiDataset(data, tokenizer, extractor, config.max_len)
