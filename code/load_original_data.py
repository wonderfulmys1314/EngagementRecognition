# -*- coding:utf-8 -*-
import numpy as np

# 加载数据
def load_data():
    # 所有标签
    train_label = np.load("../original_data/Train/train_label.npy")
    validation_label = np.load("../original_data/Validation/train_label.npy")
    test_label = np.load("../original_data/Test/train_label.npy")

    # 凝视特征
    gaze_train = np.load("../original_data/Train/gaze_array.npy")
    gaze_validation = np.load("../original_data/Validation/gaze_array.npy")
    gaze_test = np.load("../original_data/Test/gaze_array.npy")

    # 眼睛坐标
    eye_train = np.load("../original_data/Train/eye_array.npy")
    eye_validation = np.load("../original_data/Validation/eye_array.npy")
    eye_test = np.load("../original_data/Test/eye_array.npy")

    # 头部坐标
    head_train = np.load("../original_data/Train/head_array.npy")
    head_validation = np.load("../original_data/Validation/head_array.npy")
    head_test = np.load("../original_data/Test/head_array.npy")

    # 面部坐标
    face_train = np.load("../original_data/Train/face_array.npy")
    face_validation = np.load("../original_data/Validation/face_array.npy")
    face_test = np.load("../original_data/Test/face_array.npy")

    # 非刚性面部
    non_ridge_train = np.load("../original_data/Train/non_ridge_array.npy")
    non_ridge_validation = np.load("../original_data/Validation/non_ridge_array.npy")
    non_ridge_test = np.load("../original_data/Test/non_ridge_array.npy")

    # alphapose人体姿势数据
    # alphapose_train = np.load("../original_data/Train/num_array.npy")
    # alphapose_validation = np.load("../original_data/Validation/num_array.npy")
    # alphapose_test = np.load("../original_data/Test/num_array.npy")

    # 面部行为单元强度（未标准化）
    itense_train = np.load("../original_data/Train/intense_array.npy")
    itense_validation = np.load("../original_data/Validation/intense_array.npy")
    itense_test = np.load("../original_data/Test/intense_array.npy")

    # 融合特征
    X_train = np.concatenate([gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train], axis=2)
    del gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train
    X_validation = np.concatenate(
        [gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation],
        axis=2)
    del gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation
    X_test = np.concatenate([gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test], axis=2)
    del gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test

    # 合并数据集
    X_all = np.concatenate([X_train, X_validation, X_test], axis=0)
    del X_train, X_validation, X_test
    Y_all = np.concatenate([train_label, validation_label, test_label], axis=0)
    del train_label, validation_label, test_label
    return X_all, Y_all