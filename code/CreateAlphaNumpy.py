# -*- coding:utf-8 -*-

import pandas as pd
import re
import os
import numpy as np
from sklearn import preprocessing

location = "Test"
csv_path = "../AlphaPoseData/" + location
train_path = "../dataset/Labels/" + location + "Labels.csv"

getName = re.compile("(\d+)\.")


# 数值类型特征
num_columns = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y',
               'LShoulder_x', 'LShoulder_y', 'REye_x', 'REye_y', 'LEye_x', 'LEye_y',
               'REar_x', 'REar_y', 'LEar_x', 'LEar_y']


num_list = []
a = len(num_columns)
train_label = pd.read_csv(train_path)
forget = []

for i in range(len(train_label)):
    video_name = train_label.loc[i, 'ClipID']
    name = getName.findall(video_name)[0]
    # path = csv_path + "/" + name[:6] + "/" + name + "/" + video_name
    path = csv_path + "/" + name + ".csv"
    if not os.path.isfile(path):
        forget.append(name)
    else:
        print(i)
        # 读取数据
        df = pd.read_csv(path)
        # 列名
        m = list(df.columns)
        # 类型转换
        df.astype("float", inplace=True)
        # 数据长度
        length = len(df)
        # 填补差额
        if length < 300:
            extra_length = 300 - length
            add_df = [df.iloc[-1, :].values] * extra_length
            new_df = pd.DataFrame(add_df, columns=m)
            df = pd.concat([df, new_df], axis=0).reset_index(drop=True)
        # 读取数值类型数据，进行标准化处理
        num_feature = preprocessing.scale(df.loc[:299, num_columns])
        assert num_feature.shape == (300, a), print(i)
        num_list.append(num_feature)
        # 读取强度特征
        del df

num_array = np.array(num_list)
print(num_array.shape)
np.save("../mys_numpy/Test/num_array.npy", num_array)
del num_array
print(forget)
