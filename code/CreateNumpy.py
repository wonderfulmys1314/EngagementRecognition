# -*- coding:utf-8 -*-

import pandas as pd
import re
import os
import numpy as np
from sklearn import preprocessing

"""
    读取train所有文件，存放到一起，再进行数据分析
    ../dataset
"""

location = "Train"
csv_path = "../OpenFaceData/" + location
train_path = "../dataset/Labels/" + location + "Labels.csv"
store_path = "/scaled_data/"


getName = re.compile("(\d+)\.")


# 数值类型特征
gaze_columns = [' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' gaze_angle_x',
                  ' gaze_angle_y']
eye_columns = [' eye_lmk_X_0', ' eye_lmk_X_1', ' eye_lmk_X_2', ' eye_lmk_X_3', ' eye_lmk_X_4',
                ' eye_lmk_X_5', ' eye_lmk_X_6', ' eye_lmk_X_7', ' eye_lmk_X_8', ' eye_lmk_X_9', ' eye_lmk_X_10',
                ' eye_lmk_X_11', ' eye_lmk_X_12', ' eye_lmk_X_13', ' eye_lmk_X_14', ' eye_lmk_X_15', ' eye_lmk_X_16',
                ' eye_lmk_X_17', ' eye_lmk_X_18', ' eye_lmk_X_19', ' eye_lmk_X_20', ' eye_lmk_X_21', ' eye_lmk_X_22',
                ' eye_lmk_X_23', ' eye_lmk_X_24', ' eye_lmk_X_25', ' eye_lmk_X_26', ' eye_lmk_X_27', ' eye_lmk_X_28',
                ' eye_lmk_X_29', ' eye_lmk_X_30', ' eye_lmk_X_31', ' eye_lmk_X_32', ' eye_lmk_X_33', ' eye_lmk_X_34',
                ' eye_lmk_X_35', ' eye_lmk_X_36', ' eye_lmk_X_37', ' eye_lmk_X_38', ' eye_lmk_X_39', ' eye_lmk_X_40',
                ' eye_lmk_X_41', ' eye_lmk_X_42', ' eye_lmk_X_43', ' eye_lmk_X_44', ' eye_lmk_X_45', ' eye_lmk_X_46',
                ' eye_lmk_X_47', ' eye_lmk_X_48', ' eye_lmk_X_49', ' eye_lmk_X_50', ' eye_lmk_X_51', ' eye_lmk_X_52',
                ' eye_lmk_X_53', ' eye_lmk_X_54', ' eye_lmk_X_55', ' eye_lmk_Y_0', ' eye_lmk_Y_1', ' eye_lmk_Y_2',
                ' eye_lmk_Y_3', ' eye_lmk_Y_4', ' eye_lmk_Y_5', ' eye_lmk_Y_6', ' eye_lmk_Y_7', ' eye_lmk_Y_8',
                ' eye_lmk_Y_9', ' eye_lmk_Y_10', ' eye_lmk_Y_11', ' eye_lmk_Y_12', ' eye_lmk_Y_13', ' eye_lmk_Y_14',
                ' eye_lmk_Y_15', ' eye_lmk_Y_16', ' eye_lmk_Y_17', ' eye_lmk_Y_18', ' eye_lmk_Y_19', ' eye_lmk_Y_20',
                ' eye_lmk_Y_21', ' eye_lmk_Y_22', ' eye_lmk_Y_23', ' eye_lmk_Y_24', ' eye_lmk_Y_25', ' eye_lmk_Y_26',
                ' eye_lmk_Y_27', ' eye_lmk_Y_28', ' eye_lmk_Y_29', ' eye_lmk_Y_30', ' eye_lmk_Y_31', ' eye_lmk_Y_32',
                ' eye_lmk_Y_33', ' eye_lmk_Y_34', ' eye_lmk_Y_35', ' eye_lmk_Y_36', ' eye_lmk_Y_37', ' eye_lmk_Y_38',
                ' eye_lmk_Y_39', ' eye_lmk_Y_40', ' eye_lmk_Y_41', ' eye_lmk_Y_42', ' eye_lmk_Y_43', ' eye_lmk_Y_44',
                ' eye_lmk_Y_45', ' eye_lmk_Y_46', ' eye_lmk_Y_47', ' eye_lmk_Y_48', ' eye_lmk_Y_49', ' eye_lmk_Y_50',
                ' eye_lmk_Y_51', ' eye_lmk_Y_52', ' eye_lmk_Y_53', ' eye_lmk_Y_54', ' eye_lmk_Y_55', ' eye_lmk_Z_0',
                ' eye_lmk_Z_1', ' eye_lmk_Z_2', ' eye_lmk_Z_3', ' eye_lmk_Z_4', ' eye_lmk_Z_5', ' eye_lmk_Z_6',
                ' eye_lmk_Z_7', ' eye_lmk_Z_8', ' eye_lmk_Z_9', ' eye_lmk_Z_10', ' eye_lmk_Z_11', ' eye_lmk_Z_12',
                ' eye_lmk_Z_13', ' eye_lmk_Z_14', ' eye_lmk_Z_15', ' eye_lmk_Z_16', ' eye_lmk_Z_17', ' eye_lmk_Z_18',
                ' eye_lmk_Z_19', ' eye_lmk_Z_20', ' eye_lmk_Z_21', ' eye_lmk_Z_22', ' eye_lmk_Z_23', ' eye_lmk_Z_24',
                ' eye_lmk_Z_25', ' eye_lmk_Z_26', ' eye_lmk_Z_27', ' eye_lmk_Z_28', ' eye_lmk_Z_29', ' eye_lmk_Z_30',
                ' eye_lmk_Z_31', ' eye_lmk_Z_32', ' eye_lmk_Z_33', ' eye_lmk_Z_34', ' eye_lmk_Z_35', ' eye_lmk_Z_36',
                ' eye_lmk_Z_37', ' eye_lmk_Z_38', ' eye_lmk_Z_39', ' eye_lmk_Z_40', ' eye_lmk_Z_41', ' eye_lmk_Z_42',
                ' eye_lmk_Z_43', ' eye_lmk_Z_44', ' eye_lmk_Z_45', ' eye_lmk_Z_46', ' eye_lmk_Z_47', ' eye_lmk_Z_48',
                ' eye_lmk_Z_49', ' eye_lmk_Z_50', ' eye_lmk_Z_51', ' eye_lmk_Z_52', ' eye_lmk_Z_53', ' eye_lmk_Z_54',
                ' eye_lmk_Z_55']
head_columns = [' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz']
face_columns = [' X_0', ' X_1', ' X_2', ' X_3', ' X_4', ' X_5', ' X_6', ' X_7', ' X_8', ' X_9', ' X_10', ' X_11', ' X_12',
                ' X_13', ' X_14', ' X_15', ' X_16', ' X_17', ' X_18', ' X_19', ' X_20', ' X_21', ' X_22', ' X_23',
                ' X_24', ' X_25', ' X_26', ' X_27', ' X_28', ' X_29', ' X_30', ' X_31', ' X_32', ' X_33', ' X_34',
                ' X_35', ' X_36', ' X_37', ' X_38', ' X_39', ' X_40', ' X_41', ' X_42', ' X_43', ' X_44', ' X_45',
                ' X_46', ' X_47', ' X_48', ' X_49', ' X_50', ' X_51', ' X_52', ' X_53', ' X_54', ' X_55', ' X_56',
                ' X_57', ' X_58', ' X_59', ' X_60', ' X_61', ' X_62', ' X_63', ' X_64', ' X_65', ' X_66', ' X_67',
                ' Y_0', ' Y_1', ' Y_2', ' Y_3', ' Y_4', ' Y_5', ' Y_6', ' Y_7', ' Y_8', ' Y_9', ' Y_10', ' Y_11',
                ' Y_12', ' Y_13', ' Y_14', ' Y_15', ' Y_16', ' Y_17', ' Y_18', ' Y_19', ' Y_20', ' Y_21', ' Y_22',
                ' Y_23', ' Y_24', ' Y_25', ' Y_26', ' Y_27', ' Y_28', ' Y_29', ' Y_30', ' Y_31', ' Y_32', ' Y_33',
                ' Y_34', ' Y_35', ' Y_36', ' Y_37', ' Y_38', ' Y_39', ' Y_40', ' Y_41', ' Y_42', ' Y_43', ' Y_44',
                ' Y_45', ' Y_46', ' Y_47', ' Y_48', ' Y_49', ' Y_50', ' Y_51', ' Y_52', ' Y_53', ' Y_54', ' Y_55',
                ' Y_56', ' Y_57', ' Y_58', ' Y_59', ' Y_60', ' Y_61', ' Y_62', ' Y_63', ' Y_64', ' Y_65', ' Y_66',
                ' Y_67', ' Z_0', ' Z_1', ' Z_2', ' Z_3', ' Z_4', ' Z_5', ' Z_6', ' Z_7', ' Z_8', ' Z_9', ' Z_10',
                ' Z_11', ' Z_12', ' Z_13', ' Z_14', ' Z_15', ' Z_16', ' Z_17', ' Z_18', ' Z_19', ' Z_20', ' Z_21',
                ' Z_22', ' Z_23', ' Z_24', ' Z_25', ' Z_26', ' Z_27', ' Z_28', ' Z_29', ' Z_30', ' Z_31', ' Z_32',
                ' Z_33', ' Z_34', ' Z_35', ' Z_36', ' Z_37', ' Z_38', ' Z_39', ' Z_40', ' Z_41', ' Z_42', ' Z_43',
                ' Z_44', ' Z_45', ' Z_46', ' Z_47', ' Z_48', ' Z_49', ' Z_50', ' Z_51', ' Z_52', ' Z_53', ' Z_54',
                ' Z_55', ' Z_56', ' Z_57', ' Z_58', ' Z_59', ' Z_60', ' Z_61', ' Z_62', ' Z_63', ' Z_64', ' Z_65',
                ' Z_66', ' Z_67']
non_ridge_columns = [' p_scale', ' p_rx', ' p_ry', ' p_rz', ' p_tx', ' p_ty', ' p_0', ' p_1', ' p_2',
                     ' p_3', ' p_4', ' p_5', ' p_6', ' p_7', ' p_8', ' p_9', ' p_10', ' p_11', ' p_12', ' p_13', ' p_14',
                     ' p_15', ' p_16', ' p_17', ' p_18', ' p_19', ' p_20', ' p_21', ' p_22', ' p_23', ' p_24', ' p_25',
                     ' p_26', ' p_27', ' p_28', ' p_29', ' p_30', ' p_31', ' p_32', ' p_33']

# 强度特征
intense_columns = [' AU01_r', ' AU02_r',
                  ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r',
                  ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
# 二分类特征
binary_columns = [' AU01_c', ' AU02_c', ' AU04_c',
                  ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c',
                  ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']


# 眼睛上下特征
eye_subtract = [(' eye_lmk_Y_1', ' eye_lmk_Y_7'), (' eye_lmk_Y_2', ' eye_lmk_Y_6'), (' eye_lmk_Y_3', ' eye_lmk_Y_5'), (' eye_lmk_Y_9', ' eye_lmk_Y_19'),
(' eye_lmk_Y_10', ' eye_lmk_Y_18'), (' eye_lmk_Y_11', ' eye_lmk_Y_17'), (' eye_lmk_Y_12', ' eye_lmk_Y_16'), (' eye_lmk_Y_13', ' eye_lmk_Y_15'), (' eye_lmk_Y_29', ' eye_lmk_Y_35'),
(' eye_lmk_Y_30', ' eye_lmk_Y_34'), (' eye_lmk_Y_31', ' eye_lmk_Y_33'), (' eye_lmk_Y_37', ' eye_lmk_Y_47'), (' eye_lmk_Y_38', ' eye_lmk_Y_46'), (' eye_lmk_Y_39', ' eye_lmk_Y_45'),
(' eye_lmk_Y_40', ' eye_lmk_Y_44'), (' eye_lmk_Y_41', ' eye_lmk_Y_43')]

# 脸部特征
# face_substract = []

eye_subtract_columns = []

gaze_list = []
face_list = []
head_list = []
non_ridge_list = []
eye_list = []
label_list = []
intense_list = []
binary_list = []
eye_subtract_list = []
train_label = pd.read_csv(train_path)
forget = []


for i in range(len(train_label)):
    print(i)
    video_name = train_label.loc[i, 'ClipID']
    name = getName.findall(video_name)[0]
    path = csv_path + "/" + name + ".csv"
    if not os.path.isfile(path):
        forget.append(name)
    else:
        # 添加进标签
        label_list.append(train_label.loc[i, 'Engagement'])
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
        # 读取凝视类型数据，进行标准化处理
        gaze_feature = preprocessing.scale(df.loc[:299, gaze_columns])
        gaze_list.append(gaze_feature)
        # 读取眼睛数据
        eye_feature = preprocessing.scale(df.loc[:299, eye_columns])
        eye_list.append(eye_feature)
        # 读取头部特征
        head_feature = preprocessing.scale(df.loc[:299, head_columns])
        head_list.append(head_feature)
        # 读取脸部特征
        face_feature = preprocessing.scale(df.loc[:299, face_columns])
        face_list.append(face_feature)
        # 读取非刚性面部特征
        non_ridge_feature = preprocessing.scale(df.loc[:299, non_ridge_columns])
        non_ridge_list.append(non_ridge_feature)
        # 读取强度特征
        intense_feature = preprocessing.scale(df.loc[:299, intense_columns])
        intense_list.append(intense_feature)
        # 读取二分类特征
        binary_feature = np.array(df.loc[:299, binary_columns])
        binary_list.append(binary_feature)
        # 眼睛上下特征变化
        a = []
        for feature_1, feature_2 in eye_subtract:
            # eye_subtract_columns.append(feature_1 + "_" + feature_2)
            b = np.array(df[feature_1] - df[feature_2])[0:300]
            a.append(b)        
        a = np.reshape(np.array(a), (300, -1))
        eye_subtract_list.append(a)
        print(len(eye_subtract_list))
        del df

# 凝视特征
gaze_array = np.array(gaze_list)
print(gaze_array.shape)
np.save(".." + store_path + location + "/gaze_array.npy", gaze_array)
del gaze_array
# 眼部特征
eye_array = np.array(eye_list)
print(eye_array.shape)
np.save(".." + store_path + location + "/eye_array.npy", eye_array)
del eye_array
# 头部特征
head_array = np.array(head_list)
print(head_array.shape)
np.save(".." + store_path + location + "/head_array.npy", head_array)
del head_array
# 脸部特征
face_array = np.array(face_list)
print(face_array.shape)
np.save(".." + store_path + location + "/face_array.npy", face_array)
del face_array
# 非刚性面部
non_ridge_array = np.array(non_ridge_list)
print(non_ridge_array.shape)
np.save(".." + store_path + location + "/non_ridge_array.npy", non_ridge_array)
del non_ridge_array
# 强度
intense_array = np.array(intense_list)
print(intense_array.shape)
np.save(".." + store_path + location + "/intense_array.npy", intense_array)
del intense_array
# 二分类
binary_array = np.array(binary_list)
print(binary_array.shape)
np.save(".." + store_path + location + "/binary_array.npy", binary_array)
del binary_array
# 标签
label_array = np.array(label_list)
print(label_array.shape)
np.save(".." + store_path + location + "/train_label.npy", label_array)
del label_array
eye_subtract_array = np.array(eye_subtract_list)
print(eye_subtract_array.shape)
np.save(".." + store_path+ location + "/eye_subtract_array.npy", eye_subtract_array)
del eye_subtract_array
