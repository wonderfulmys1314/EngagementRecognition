import keras
import tensorflow as tf
from keras.layers import *
from keras.layers.convolutional import Conv1D,UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2
from keras.models import *
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam, Adagrad, SGD, Nadam
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import LSTM, Bidirectional, GRU, TimeDistributed, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler

from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from keras.callbacks import EarlyStopping  
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import os 
import math
import argparse


# 随机数设置
tf.set_random_seed(seed=2019)
np.random.seed(2019)


# 所有标签
train_label = np.load("../mys_numpy/Train/train_label.npy")
validation_label = np.load("../mys_numpy/Validation/train_label.npy")
test_label = np.load("../mys_numpy/Test/train_label.npy")

# 凝视特征
gaze_train = np.load("../mys_numpy/Train/gaze_array.npy")
gaze_validation = np.load("../mys_numpy/Validation/gaze_array.npy")
gaze_test = np.load("../mys_numpy/Test/gaze_array.npy")

# 眼睛坐标
eye_train = np.load("../mys_numpy/Train/eye_array.npy")
eye_validation = np.load("../mys_numpy/Validation/eye_array.npy")
eye_test = np.load("../mys_numpy/Test/eye_array.npy")

# 头部坐标
head_train = np.load("../mys_numpy/Train/head_array.npy")
head_validation = np.load("../mys_numpy/Validation/head_array.npy")
head_test = np.load("../mys_numpy/Test/head_array.npy")

# 面部坐标
face_train = np.load("../mys_numpy/Train/face_array.npy")
face_validation = np.load("../mys_numpy/Validation/face_array.npy")
face_test = np.load("../mys_numpy/Test/face_array.npy")

# 非刚性面部
non_ridge_train = np.load("../mys_numpy/Train/non_ridge_array.npy")
non_ridge_validation = np.load("../mys_numpy/Validation/non_ridge_array.npy")
non_ridge_test = np.load("../mys_numpy/Test/non_ridge_array.npy")

# alphapose人体姿势数据
alphapose_train = np.load("../mys_numpy/Train/num_array.npy")
alphapose_validation = np.load("../mys_numpy/Validation/num_array.npy")
alphapose_test = np.load("../mys_numpy/Test/num_array.npy")


# 面部行为单元强度（未标准化）
itense_train = np.load("../mys_numpy/Train/intense_array.npy")
itense_validation = np.load("../mys_numpy/Validation/intense_array.npy")
itense_test = np.load("../mys_numpy/Test/intense_array.npy")

# 融合特征
X_train = np.concatenate([gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train], axis=2)
del gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train
X_validation = np.concatenate([gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation], axis=2)
del gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation
X_test = np.concatenate([gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test], axis=2)
del gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test

# 特征数目
nb_input_vector = X_train.shape[2]

# 综合起来
X_all = np.concatenate([X_train, X_validation, X_test], axis=0)
del X_train, X_validation, X_test
Y_all = np.concatenate([train_label, validation_label, test_label], axis=0)
del train_label, validation_label, test_label

print("类比比例")
print("类别1: ", len(Y_all[Y_all == 0]) / len(Y_all))
print("类别2: ", len(Y_all[Y_all == 1]) / len(Y_all))
print("类别3: ", len(Y_all[Y_all == 2]) / len(Y_all))
print("类别4: ", len(Y_all[Y_all == 3]) / len(Y_all))

Y_all = np_utils.to_categorical(Y_all, 4)


# 切分
X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, shuffle=True, random_state=0)


# 定义学习率下降方式
# 直接下降
def descentLR(epoch, SWA_START, lr_start, lr_end):
    t = epoch / SWA_START
    lr_ratio = lr_end / lr_start
    if t<= 0.5:
        factor = 1.0
    elif t<=0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_start * factor


# 循环下降
def cyclicLR(epoch, c, lr_start, lr_end):
    if epoch<30:
        return 3e-5
    t_epoch = (epoch % c) / c
    lr_now = (1 - t_epoch) * lr_start + t_epoch * lr_end
    return lr_now


# 指数下降
def expLR(epoch, c, lr_start, lr_end):
    t_epoch = (epoch % c) / c
    lr_now = (lr_start - lr_end) * math.cos(t_epoch*math.pi/2) + lr_end
    return lr_now


# 学习率规划
def LR_schedule(epoch, flag=0, SWA_START=150, lr_start=0.1, lr_end=0.05, c=100):
    # 循环下降
    if flag == 0:
        return cyclicLR(epoch, c, lr_start, lr_end)
    # 直接下降
    elif flag == 1:
        return descentLR(epoch, SWA_START, lr_start, lr_end)
    # 指数下降
    elif flag == 2:
        return expLR(epoch, c, lr_start, lr_end)
    # 保持不变
    else:
        return lr_start


# 定义随机权重综合
class SWA(keras.callbacks.Callback):
    # 初始化
    def __init__(self, SWA_START, update):
        super(SWA, self).__init__()
        self.SWA_START = SWA_START
        self.update = update

    # 训练开始
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params["epochs"]
        print('Stochastic weight averaging selected for every {} epochs.'
              .format(self.update))

    # 轮次开始
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        print("learning_rate of current epoch is : {}".format(lr))

    # 轮次结束
    def on_epoch_end(self, epoch, logs=None):
        # 初始化权重
        if epoch == self.SWA_START:
            self.swa_weights = self.model.get_weights()

        # 学习率最小处对权重进行更新
        elif epoch > self.SWA_START:
            if (epoch - self.SWA_START)%self.update == 0:
                n_model = (epoch - self.SWA_START) / self.update
                for i, weights in enumerate(self.model.get_weights()):
                    self.swa_weights[i] = (self.swa_weights[i] * n_model + weights) / (n_model + 1)
        else:
            pass

    # 训练结束，保留学习到的模型参数
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('set stochastic weight average as final model parameters [FINISH].')


# 设计专注度层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


# 权重惩罚
def weight_categorical_crossentropy(y_true, y_pred):
    # 权重惩罚和类别保持一致
    weight = tf.constant([1.0, 1.0, 1.0, 1.0])
    y_coe = y_true * weight
    loss1 = K.categorical_crossentropy(y_coe, y_pred)
    return loss1


# LSTM模型
def LSTM_model():
    inputs = Input((300, nb_input_vector))  # 规定输入大小
    x = BatchNormalization()(inputs)
    x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)

    x = Convolution1D(256, 5, padding="same",  use_bias=True)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = AvgPool1D(pool_size=2, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, use_bias=True, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    x = AttentionLayer()(x)
    # x = GlobalAveragePooling1D()(x)
    # x = Dropout(0.2)(x)
    x = Dense(16, use_bias=True, activation="relu")(x)
    # x = Dropout(0.2)(x)
    outputs = Dense(4, use_bias=True, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# 变量初始化
batch_size = 600
nb_classes = 4
nb_epoch = 300
swa_start = 100
update = 20
c = 30
lr_start = 2e-5
lr_end = 1e-5
result = []


# 学习率变化
schedule = lambda epoch:LR_schedule(epoch, 5, SWA_START=swa_start, lr_start=lr_start, lr_end=lr_end, c=c)
lr_schedule_obj = LearningRateScheduler(schedule=schedule)
# 随机权重对象
swa_obj = SWA(swa_start, update)
# 记录对象
history = LossHistory()
model = LSTM_model()
model.compile(optimizer=RMSprop(lr=lr_start, rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=1), loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[history, swa_obj, lr_schedule_obj])
score = model.evaluate(X_test, Y_test, verbose=0)


# 模型评估
print(result)
output = pd.DataFrame()
output["result"] = result
output.to_csv("result.csv")


# 绘制acc-loss曲线
# history.loss_plot('epoch')