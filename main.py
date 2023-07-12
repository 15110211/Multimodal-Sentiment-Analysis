# coding = utf-8
# -*- coding:utf-8 -*-
import os
import argparse
import config
from util import PreData
import model

config.setup_seed()


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--model', type=str, default='img_and_text', help='模型选择：img, text, img_and_text')
    parse.add_argument('--train', action='store_true', help='模型训练')
    parse.add_argument('--predict', action='store_true', help='模型预测')

    parse.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parse.add_argument('--epoch', type=int, default=config.epoch, help='epoch')
    parse.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parse.add_argument('--text_lr', type=float, default=config.text_lr, help='bert模型 learning rate')
    parse.add_argument('--img_lr', type=str, default=config.img_lr, help='vit模型 learning rate')

    return parse.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 根据传入命令更改config配置
    config.batch_size = args.batch_size
    config.epoch = args.epoch
    config.lr = args.lr
    config.text_lr = args.text_lr
    config.img_lr = args.img_lr

    # 数据预处理
    if (not os.path.exists(config.train_combined_path)) or (not os.path.exists(config.test_combined_path)):
        PreData.run()

    if args.model == 'img_and_text':
        if args.train:
            model.multi_train()
        if args.predict:
            model.multi_predict()
    elif args.model == 'img':
        if args.train:
            model.img_train()
        if args.predict:
            model.img_predict()
    elif args.model == 'text':
        if args.train:
            model.text_train()
        if args.predict:
            model.text_predict()
