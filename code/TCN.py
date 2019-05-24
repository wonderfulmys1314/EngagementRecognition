"""
    加载全部数据461维，减少了2D眼睛像素、2D脸部像素
    试试效果
"""
import keras
import tensorflow as tf
from keras.layers import Input, Concatenate, Flatten,Dense, Conv1D, Dropout, Activation, Reshape, BatchNormalization, Convolution1D
from keras.layers.convolutional import Conv1D,UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Dense, LSTM, Bidirectional, GRU, TimeDistributed, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.models import Sequential

from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping  
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd



tf.set_random_seed(seed=2019)
np.random.seed(2019)
label_classes = 4

train_label = np.load("/home/server/engagement/mys_numpy/Train/train_label.npy")
validation_label = np.load("/home/server/engagement/mys_numpy/Validation/train_label.npy")
test_label = np.load("/home/server/engagement/mys_numpy/Test/train_label.npy")

# the data, shuffled and split between train and test sets
# 凝视
gaze_train = np.load("/home/server/engagement/mys_numpy/Train/gaze_array.npy")
gaze_validation = np.load("/home/server/engagement/mys_numpy/Validation/gaze_array.npy")
gaze_test = np.load("/home/server/engagement/mys_numpy/Test/gaze_array.npy")

# 眼睛
eye_train = np.load("/home/server/engagement/mys_numpy/Train/eye_array.npy")
eye_validation = np.load("/home/server/engagement/mys_numpy/Validation/eye_array.npy")
eye_test = np.load("/home/server/engagement/mys_numpy/Test/eye_array.npy")

# 头部
head_train = np.load("/home/server/engagement/mys_numpy/Train/head_array.npy")
head_validation = np.load("/home/server/engagement/mys_numpy/Validation/head_array.npy")
head_test = np.load("/home/server/engagement/mys_numpy/Test/head_array.npy")

# 面部
face_train = np.load("/home/server/engagement/mys_numpy/Train/face_array.npy")
face_validation = np.load("/home/server/engagement/mys_numpy/Validation/face_array.npy")
face_test = np.load("/home/server/engagement/mys_numpy/Test/face_array.npy")

# 非刚性面部
non_ridge_train = np.load("/home/server/engagement/mys_numpy/Train/non_ridge_array.npy")
non_ridge_validation = np.load("/home/server/engagement/mys_numpy/Validation/non_ridge_array.npy")
non_ridge_test = np.load("/home/server/engagement/mys_numpy/Test/non_ridge_array.npy")

# alphapose数据
# alphapose_train = np.load("/home/server/engagement/alphapose_numpy/Train/num_array.npy")
# alphapose_validation = np.load("/home/server/engagement/alphapose_numpy/Validation/num_array.npy")
# alphapose_test = np.load("/home/server/engagement/alphapose_numpy/Test/num_array.npy")
# #

# 强度（未标准化）
# itense_train = np.load("/home/server/engagement/mys_numpy/Train/intense_array.npy")
# itense_validation = np.load("/home/server/engagement/mys_numpy/Validation/intense_array.npy")
# itense_test = np.load("/home/server/engagement/mys_numpy/Test/intense_array.npy")

# # 分类
# binary_train = np.load("/home/server/engagement/mys_numpy/Train/binary_array.npy")
# binary_validation = np.load("/home/server/engagement/mys_numpy/Validation/binary_array.npy")
# binary_test = np.load("/home/server/engagement/mys_numpy/Test/binary_array.npy")

# # 眼部相减
# eye_subtract_train = np.load("/home/server/engagement/mys_numpy/Train/eye_subtract_array.npy")
# eye_subtract_validation = np.load("/home/server/engagement/mys_numpy/Validation/eye_subtract_array.npy")
# eye_subtract_test = np.load("/home/server/engagement/mys_numpy/Test/eye_subtract_array.npy")

# merge
X_train = np.concatenate([gaze_train, eye_train,  face_train, non_ridge_train], axis=2)
X_validation = np.concatenate([gaze_validation, eye_validation, face_validation, non_ridge_validation], axis=2)
X_test = np.concatenate([gaze_test, eye_test, face_test, non_ridge_test], axis=2)


Y_train = np_utils.to_categorical(train_label, label_classes)
Y_validation = np_utils.to_categorical(validation_label, label_classes)
Y_test = np_utils.to_categorical(test_label, label_classes)
print(Y_test)

face_time_steps = X_train.shape[1]  # 时间序列长度
nb_input_vector = X_train.shape[2] # 输入序列
# i_input = itense_test.shape[2]


def ED_TCN():
    inputs = Input((300, nb_input_vector))  # 规定输入大小
    label_classes = 4
    # Encoder
    x = Conv1D(128, 45, strides=1, padding='same', use_bias=True)(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Conv1D(160, 45, strides=1, padding='same', use_bias=True)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Decoder
    x = UpSampling1D(size=2)(x)
    x = Conv1D(160, 45, strides=1, padding='same', use_bias=True)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = UpSampling1D(size=2)(x)
    x = Conv1D(128, 45, strides=1, padding='same', use_bias=True)(x)
    x = LeakyReLU(alpha=0.3)(x)

    # combine
    x = Conv1D(4, 1, strides=1, padding='valid', use_bias=True)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Activation("softmax")(x)
    outputs = GlobalAveragePooling1D()(x)
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
nb_epoch = 500

# #编译模型
ED_TCN_model = ED_TCN()


def weight_categorical_crossentropy(y_true, y_pred, e=0.1):
    weight = tf.constant(value=[1.0, 1.0, 1.0, 1.0])
    y_coe = y_true * weight
    loss1 = K.categorical_crossentropy(y_coe, y_pred)
    # loss1=-tf.reduce_mean(y_coe*tf.log(y_pred))#带权重的交叉熵
    # loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    # return (1-e)*loss1 + e*loss2
    return loss1


ED_TCN_model.compile(optimizer=RMSprop(lr=0.000001, rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=1), loss="categorical_crossentropy",metrics=['accuracy'])

# 创建一个实例history
history = LossHistory()

# 迭代训练（注意这个地方要加入callbacks）
ED_TCN_model.fit(X_train, Y_train,
                 batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1,
                 validation_data=(X_test, Y_test),
                 callbacks=[history])

# 模型评估
score = ED_TCN_model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 绘制acc-loss曲线
history.loss_plot('epoch')
