import numpy as np
import os
from PIL import Image
import random
import pandas as pd
import time
# 10分钟
start_time = time.time()
# train.csv中抽取的训练集
train1 = {}
# test.csv中抽取的训练集
train2 = {}
# val.csv中抽取的训练集
train3 = {}
# train.csv中抽取的测试集
test1 = {}
# test.csv中抽取的测试集
test2 = {}
# val.csv中抽取的测试集
test3 = {}

# 读取图片的csv文件
train_csv = pd.read_csv("./data/train.csv")
test_csv = pd.read_csv("./data/test.csv")
val_csv = pd.read_csv("./data/val.csv")
# 取总共数据的四分之一，减小训练压力
train_numbers = int(train_csv.shape[0] * 0.25)
# 每一类共有600张图片，获取总的图片类别数
train_sorts = int(train_numbers / 600)
for s in range(1, train_sorts + 1):
    # 每一类随机取500张图片作训练
    ind = random.sample(range((s - 1) * 600, s * 600), 500)
    for i in ind:
        train1[train_csv.loc[i][0]] = train_csv.loc[i][1]
    # 剩下的100张图片用作测试
    for j in range((s - 1) * 600, s * 600):
        if j not in ind:
            test1[train_csv.loc[j][0]] = train_csv.loc[j][1]

test_numbers = int(test_csv.shape[0] * 0.25)
test_sorts = int(test_numbers / 600)
for s in range(1, test_sorts + 1):
    ind = random.sample(range((s - 1) * 600, s * 600), 500)
    for i in ind:
        train2[test_csv.loc[i][0]] = test_csv.loc[i][1]
    for j in range((s - 1) * 600, s * 600):
        if j not in ind:
            test2[test_csv.loc[j][0]] = test_csv.loc[j][1]

val_numbers = int(val_csv.shape[0] * 0.25)
val_sorts = int(val_numbers / 600)
for s in range(1, val_sorts + 1):
    ind = random.sample(range((s - 1) * 600, s * 600), 500)
    for i in ind:
        train3[val_csv.loc[i][0]] = val_csv.loc[i][1]
    for j in range((s - 1) * 600, s * 600):
        if j not in ind:
            test3[val_csv.loc[j][0]] = val_csv.loc[j][1]

train1.update(train2)
train1.update(train3)
# 训练集合并到train1
print(len(train1))
test1.update(test2)
test1.update(test3)
# 测试集合并到test1
print(len(test1))
img_path = "E:/研究生生涯论文/mini-imagenet/images"
for pic in os.listdir(img_path):
    path = img_path + '/' + pic
    im = Image.open(path)
    if pic in train1.keys():
        # 获取图片所属的类别
        tmp = train1[pic]
        temp_path = os.getcwd() + '/data' + '/train' + '/' + tmp
        if os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
        # resample表示改变图像过程中的插值方法，0表示双线性插值， 1表示最邻近插值，2表示双三次插值， 3表示面积插值法
        im_deal = im.resize((224, 224), resample=2)
        t = temp_path + '/' + pic
        # 保存图片到指定文件夹
        im_deal.save(t)
for pic in os.listdir(img_path):
    path = img_path + '/' + pic
    im = Image.open(path)
    if pic in test1.keys():
        tmp = test1[pic]
        temp_path = os.getcwd() + '/data' + '/test' + '/' + tmp
        if os.path.exists(temp_path) is False:
            os.makedirs(temp_path)
        im_deal = im.resize((224, 224), resample=2)
        t = temp_path + '/' + pic
        im_deal.save(t)
end_time = time.time()
print("总共运行时间：", (end_time - start_time))
