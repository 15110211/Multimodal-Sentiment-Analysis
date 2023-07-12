# 多模态情感分析

This is the multimodal sentiment analysis project repository for the DaSE 2023 AI course.

## Setup

This implemetation is based on Python3.9. To run the code, you need the following dependencies:

* numpy==1.23.5
* Pillow==10.0.0
* torch==1.11.0
* transformers==4.19.3

You can simply run

```bash
pip install -r requirements.txt
```

## Repository structure

I select some important files for detailed description.

```
|-- cache/                              # 缓存
    |-- model                           # 训练模型缓存
    |-- prediction.txt                  # 预测结果缓存
|-- data/                               # 数据
    |-- input/                          # 输入数据
        |-- test_data.json              # 预处理后的测试数据
        |-- train_data.json             # 预处理后的训练数据
    |-- data/                            # 名字为id的所有图片和文本数据
    |-- test_without_label.txt          # 原始数据
    |-- train.txt                       # 原始数据
|-- util/                               # 相关定义文件
    |-- PreData.py                      # 数据预处理文件
    |-- imgDataset.py                   # 图像数据集相关
    |-- multiDataset.py                 # 多模态数据集相关
    |-- textDataset.py                  # 文本数据集相关
|-- config.py                           # 配置文件
|-- model.py                            # 多模态以及单模态模型
|-- main.py                             # 运行主函数
|-- runUtil.py                          # 工具方法
```

## Run pipeline

1. Place all data in the right directory('./data/data'). 

2. Run `main.py` . You can run any models implemented in 'models.py'. Including multimodal model, text single-modal model, IMG single-modal model.

   Train the model

   ```bash
   python main.py --model img_and_text --train
   ```

   Predict

   ```bash
   python main.py --model img_and_text --predict
   ```

   A list of parameters that can be specified

   |           参数            |              默认值                |                   说明                  |
   | :-----------------------: | :-------------------------------: | :-------------------------------------: |
   |          --model          |           img_and_text            | 实现模型类型：img, text, img_and_text    |
   |          --train          |               False               |                   模型训练               |
   |         --predict         |               False               |                   模型预测               |
   |       --batch_size        |                32                 |                  batch size             |
   |          --epoch          |                 5                 |                    epoch                |
   |           --lr            |               1e-3                |               多模态模型学习率            |
   |         --text_lr         |               2e-5                |               text模型 bert学习率        |
   |         --img_lr          |               2e-5                |                img模型 vit学习率         |


## Reference

1. Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee"ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks" arXiv:1908.02265(2019)

2. Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai"VL-BERT: Pre-training of Generic Visual-Linguistic Representations" arXiv:1908.08530(2019)

3. Austin Reiter, Menglin Jia, Pu Yang, Ser-Nam Lim"Deep Multi-Modal Sets" arXiv:2003.01607(2020)

4. Joint Fine-Tuning for Multimodal Sentiment Analysis：guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis: This is the code for the Paper "Guilherme L. Toledo, Ricardo M. Marcacini: Transfer Learning with Joint Fine-Tuning for Multimodal Sentiment Analysis (LXAI Research Workshop at ICML 2022)". (github.com)
