# 获取cifar10数据
# 测试的时候可以自己拍得到实际的照片来测试
class CifarDataReader():
    import pickle
    import numpy as np
    import os
    import random
    def __init__(self, cifar_folder, onehot=False, file_number=1):
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.train_data = self.read_train_file()
        self.test_data = self.read_test_data()
        self.batch_index = 0
        self.file_number = file_number

    # 读取数据,返回dict, 以字节编码方式返回
    def unpickle(self, f):
        with open(f, 'rb') as file:
            try:
                dicts = self.pickle.load(file, encoding='bytes')
            except Exception as e:
                print('load error', e)
        return dicts

    # 读取一个训练集文件，返回数据list
    def read_train_file(self, files=''):
        if files:
            files = self.os.path.join(self.cifar_folder, files)
        else:
            # files=''执行else语句
            # os.path.join函数用来连接两个或更多个路径名组件，路径之间自动加上分隔符
            files = self.os.path.join(self.cifar_folder, 'data_batch_%d' % self.file_number)
        dict_train = self.unpickle(files)
        # 将数据和对应标签数据打包
        train_data = list(zip(dict_train[b'data'], dict_train[b'labels']))
        self.np.random.shuffle(train_data)
        print('成功读取到训练集数据：data_batch_%d' % self.file_number)
        return train_data

    # 读取测试集数据
    def read_test_data(self):
        files = self.os.path.join(self.cifar_folder, 'test_batch')
        dict_test = self.unpickle(files)
        test_data = list(zip(dict_test[b'data'], dict_test[b'labels']))  # 将数据和对应标签打包
        print('成功读取测试集数据')
        return test_data










