# coding = utf-8
# -*- coding:utf-8 -*-
import sys
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import ViTModel, BertModel

import config
from runUtil import train, test, predict, device
from util.multiDataset import getMultiDataset
from util.imgDataset import getImgDataset
from util.textDataset import getTextDataset

sys.path.append(config.root_path)

config.setup_seed()


class MultiModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.vit = ViTModel.from_pretrained(config.pretrained_model_img,config=config.pretrained_model_img)
        for param in self.vit.parameters():
            param.requires_grad_(fine_tune)

        self.bert = BertModel.from_pretrained(config.pretrained_model_text,config=config.pretrained_model_text)
        for param in self.bert.parameters():
            param.requires_grad_(fine_tune)

        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(768 * 2, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        img_out = self.vit(pixel_values=pixel_values)
        bert_out = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out = torch.concat([img_out['pooler_output'], bert_out['pooler_output']], dim=1)
        out = self.fc(self.drop(out))

        return out


class TextModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.pretrained_model_text,config=config.pretrained_model_text)
        for param in self.bert.parameters():
            param.requires_grad_(fine_tune)

        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        out = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out = self.fc(self.drop(out['pooler_output']))
        return out
    

class ImgModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.vit = ViTModel.from_pretrained(config.pretrained_model_img,config=config.pretrained_model_img)
        for param in self.vit.parameters():
            param.requires_grad_(fine_tune)

        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)
        out = self.vit(pixel_values=pixel_values)
        out = self.fc(self.drop(out['pooler_output']))
        return out


def multi_train():
    model = MultiModel(fine_tune=True)
    model.to(device)

    bert_params = list(map(id, model.bert.parameters()))
    vit_params = list(map(id, model.vit.parameters()))
    down_params = filter(lambda p: id(p) not in bert_params + vit_params, model.parameters())
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': config.bert_lr},
        {'params': model.vit.parameters(), 'lr': config.img_lr},
        {'params': down_params, 'lr': config.lr}
    ])

    dataset = getMultiDataset(config.train_combined_path)
    train_dataset = Subset(dataset, range(0, 3200))
    val_dataset = Subset(dataset, range(3200, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader)
    model = torch.load(config.cache_model_path, map_location=device)
    print('Congratulation！Final accuracy:', test(model, val_loader))


def multi_predict():
    model = torch.load(config.cache_model_path, map_location=device)
    test_loader = DataLoader(getMultiDataset(config.test_combined_path), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


def text_train():
    model = TextModel(fine_tune=True)
    model.to(device)

    bert_params = list(map(id, model.bert.parameters()))
    down_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': config.bert_lr},
        {'params': down_params, 'lr': config.lr}
    ])

    dataset = getTextDataset(config.train_combined_path)
    train_dataset = Subset(dataset, range(0, 3200))
    val_dataset = Subset(dataset, range(3200, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader)
    model = torch.load(config.cache_model_path, map_location=device)
    print('Congratulation！Final accuracy:', test(model, val_loader))


def text_predict():
    model = torch.load(config.cache_model_path, map_location=device)
    test_loader = DataLoader(getTextDataset(config.test_combined_path), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


def img_train():
    model = ImgModel(fine_tune=True)
    model.to(device)

    vit_params = list(map(id, model.vit.parameters()))
    down_params = filter(lambda p: id(p) not in vit_params, model.parameters())
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': config.img_lr},
        {'params': down_params, 'lr': config.lr}
    ])

    dataset = getImgDataset(config.train_combined_path)
    train_dataset = Subset(dataset, range(0, 3200))
    val_dataset = Subset(dataset, range(3200, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader)
    model = torch.load(config.cache_model_path, map_location=device)
    print('Congratulation！Final accuracy:', test(model, val_loader))


def img_predict():
    model = torch.load(config.cache_model_path, map_location=device)
    test_loader = DataLoader(getImgDataset(config.test_combined_path), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)



if __name__ == '__main__':
    multi_train()
    multi_predict()
    text_train()
    text_predict()
    img_train()
    img_predict()
