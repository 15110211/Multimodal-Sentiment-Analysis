# coding = utf-8
# -*- coding:utf-8 -*-
import json
import os
import config

# 将train.txt、test.txt和给的文本图片结合起来
def run():
    train_data = list()
    test_data = list()
    labels = dict()

    with open(config.train_path, 'r', encoding='utf-8') as fs:
        next(fs)        
        for line in fs:
            guid, tag = line.strip().split(',')
            labels[int(guid)] = tag

    with open(config.test_without_label_path, 'r', encoding='utf-8') as fs:
        next(fs)
        for line in fs:
            guid, tag = line.strip().split(',')
            labels[int(guid)] = ''

    for root, _, files in os.walk(config.data_path):
        for f in files:
            if f[-3:] == 'txt':
                path = os.path.join(root, f)
                with open(path, 'r', encoding='ANSI') as fs:
                    text = fs.read()
                    guid = int(f[:-4])
                    tag = labels.get(guid)
                    data = {'guid': guid,'text': text.strip(),'tag': tag,'img': str(guid) + '.jpg'}
                    if tag is not None:
                        if tag != '':
                            train_data.append(data)
                        else:
                            test_data.append(data)

    print(len(train_data))
    print(len(test_data))

    # 写入数据
    with open(config.train_combined_path, 'w', encoding='utf-8') as fs:
        json.dump(train_data, fs, ensure_ascii=False)
    with open(config.test_combined_path, 'w', encoding='utf-8') as fs:
        json.dump(test_data, fs, ensure_ascii=False)


if __name__ == '__main__':
    run()
