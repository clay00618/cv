import numpy as np
import random
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
import pickle
import torch

# data = pd.read_csv("./data/train.csv")
# print(data.head(5))
# print(data.shape)
# print(data.loc[1])
# print(type(data.loc[1]))
# print(type(list(data.loc[1])))
# print(list(data.loc[1]))
# print(data.loc[1][0])

# label = np.array([1, 2, 3, 4])
# label_batch = tf.reshape(label, [4])
# print(label_batch)

# with open('./cifar-10-batches-py/data_batch_1', 'rb') as file:
#     try:
#         dicts = pickle.load(file, encoding='bytes')
#     except Exception as e:
#         print('load error', e)
# print(dicts)

char = torch.FloatTensor([[-1.1, 1, -0.2],[-1.2, -0.6, 1.2],[-1.5,1.6,-0.2],[1.4,-0.1,-0.2]])
y_test = torch.LongTensor([1, 2, 1, 0])
print(char)
print("行数：", char.shape[0])
for i in range(char.shape[0]):
    max_idx = char[i].argmax()
    print(max_idx)
    if torch.eq(max_idx, y_test[i]):
        print("equal")


