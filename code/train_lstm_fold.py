import keras
import tensorflow as tf
from keras.layers import *
from keras.layers.convolutional import Conv1D, UpSampling1D
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
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import os
import math
import argparse
from load_data import load_data

# 随机数设置
tf.set_random_seed(seed=2019)
np.random.seed(2019)

# 加载数据
X_all, Y_all = load_data()
# 获取维度
nb_input_vector = X_all.shape[2]

# 依据类别属性对类别进行修改
# 打印类别比例
print("类比比例")
print("类别1: ", len(Y_all[Y_all == 0]) / len(Y_all))
print("类别2: ", len(Y_all[Y_all == 1]) / len(Y_all))
print("类别3: ", len(Y_all[Y_all == 2]) / len(Y_all))
print("类别4: ", len(Y_all[Y_all == 3]) / len(Y_all))


# 定义学习率下降方式
# 直接下降
def descentLR(epoch, SWA_START, lr_start, lr_end):
    t = epoch / SWA_START
    lr_ratio = lr_end / lr_start
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_start * factor


# 循环下降
def cyclicLR(epoch, c, lr_start, lr_end):
    t_epoch = (epoch % c) / c
    lr_now = lr_start - (lr_start-lr_end) * t_epoch
    return lr_now


# 指数下降
def expLR(epoch, c, lr_start, lr_end):
    t_epoch = (epoch % c) / c
    lr_now = (lr_start - lr_end) * math.cos(t_epoch * math.pi / 2) + lr_end
    return lr_now


# 学习率规划
def LR_schedule(epoch, flag=0, lr_start=0.1, lr_end=0.05, c=100):
    # 循环下降
    if flag == 0:
        return cyclicLR(epoch, c, lr_start, lr_end)
    # 直接下降
    elif flag == 1:
        return descentLR(epoch, c, lr_start, lr_end)
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
            if (epoch - self.SWA_START) % self.update == 0:
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
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        x = K.permute_dimensions(inputs, (0, 2, 1))
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
    x = Dropout(0.2)(inputs)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, use_bias=True, return_sequences=True))(x)


    # x = Dense(256, bias=True, activation="tanh")(x)

    x = GlobalMaxPooling1D()(x)
    # bx = AttentionLayer()(x)
    # cx = GlobalAveragePooling1D()(x)
    # x = Concatenate()([ax, bx, cx])
    x = Dropout(0.2)(x)
    outputs = Dense(4, bias=True, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

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

    def loss_plot(self, loss_type, name):
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
        plt.savefig(str(name)+".png")


# 变量初始化
batch_size = 600
nb_classes = 4
nb_epoch = 400
swa_start = 0
update = 50
c = 40
lr_start = 4e-5
lr_end = 1e-5
result = []


# 学习率变化
schedule = lambda epoch: LR_schedule(epoch, 5, lr_start=lr_start, lr_end=lr_end, c=c)
lr_schedule_obj = LearningRateScheduler(schedule=schedule)
# 随机权重对象
swa_obj = SWA(swa_start, update)
# 记录对象
history = LossHistory()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
i = 1
for train_index, test_index in skf.split(X_all, Y_all):
    # 随机数设置
    print("开始第{0}轮验证".format(i))
    model = LSTM_model()
    model.compile(optimizer=RMSprop(rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=1),
                  loss=weight_categorical_crossentropy, metrics=['accuracy'])

    X_train = X_all[train_index]
    X_test = X_all[test_index]
    m = Y_all[train_index]
    Y_train = np_utils.to_categorical(Y_all[train_index], 4)
    Y_test = np_utils.to_categorical(Y_all[test_index], 4)

    print("校验类比比例")
    print("类别1: ", len(m[m == 0]) / len(m))
    print("类别2: ", len(m[m == 1]) / len(m))
    print("类别3: ", len(m[m == 2]) / len(m))
    print("类别4: ", len(m[m == 3]) / len(m))

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[history, swa_obj, lr_schedule_obj])

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    history.loss_plot('epoch', i)
    i += 1



