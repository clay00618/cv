import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow.python.keras

start_time = time.time()
x_train = np.ndarray([])
y_train = np.ndarray([])
x_test = np.ndarray([])
y_test = np.ndarray([])
I = np.identity(len(os.listdir(os.getcwd() + "/data" + "/train")))


with tf.Session() as sess:
    for t, name in enumerate(os.listdir(os.getcwd() + "/data" + "/train")):
        for pic in os.listdir(os.getcwd() + "/data" + "/train" + "/" + name):
            image_raw_data = tf.gfile.FastGFile(os.getcwd() + "/data" + "/train" + "/" + name + "/" + pic, 'rb').read()
            # 图片解码
            img_data = tf.image.decode_jpeg(image_raw_data)
            # 将图片转为float32
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            # 转为numpy数组
            # print(type(img_data.eval()))
            # print(sess.run(tf.shape(img_data)))
            x_train = np.append(x_train, img_data.eval().reshape(1, -1))
            y_train = np.append(y_train, I[t, :])
    print(x_train.shape)
    print(y_train.shape)

    for t, name in enumerate(os.listdir(os.getcwd() + "/data" + "/test")):
        for pic in os.listdir(os.getcwd() + "/data" + "/test" + "/" + name):
            image_raw_data = tf.gfile.FastGFile(os.getcwd() + "/data" + "/test" + "/" + name + "/" + pic, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            x_test = np.append(x_test, img_data.eval().reshape(1, -1))
            y_test = np.append(y_test, I[t, :])
    print(x_test.shape)
    print(y_test.shape)


end_time = time.time()
print("总共运行时间：", end_time - start_time)

