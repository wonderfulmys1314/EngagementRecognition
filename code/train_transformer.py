# -*- coding: utf-8 -*-

'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
# from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from modules import *
import os, codecs
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import warnings
from keras.utils import np_utils

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Graph(object):
    # 初始化
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.x = tf.placeholder(tf.int32, shape=(None, hp.length))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.labels))           

            # 编码器
            with tf.variable_scope("encoder"):
                # 位置编码
                if hp.sinusoid:
                    self.x += positional_encoding(self.x,
                                                  num_units=hp.hidden_units,
                                                  zero_pad=False,
                                                  scale=False,
                                                  scope="enc_pe")
                else:
                    self.x += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="enc_pe")

                # Dropout
                # 如果训练，则dropout
                self.x = tf.layers.dropout(self.enc, 
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))
                
                # 自身专注训练
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # 多输入专注
                        self.x = multihead_attention(queries=self.x, 
                                                     keys=self.x,
                                                     num_units=hp.hidden_units,
                                                     num_heads=hp.num_heads,
                                                     dropout_rate=hp.dropout_rate,
                                                     is_training=is_training,
                                                     causality=False)
                        
                        # Feed Forward
                        self.x = feedforward(self.x, num_units=[4*hp.hidden_units, hp.hidden_units])

            # 全局池化
            self.average_value = tf.reduce_mean(self.x, axis=1)
            self.logits = tf.layers.dense(self.average_value, hp.labels)
            self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
            self.y_true = tf.to_int32(tf.argmax(self.y, dimension=-1))
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.y_true)))
            tf.summary.scalar('acc', self.acc)

            # 如果在训练
            if is_training:  
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.mean_loss = tf.reduce_mean(self.loss)
                print("mean_loss")

               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # 所有标签
    train_label = np.load("../mys_numpy/Train/train_label.npy")
    validation_label = np.load("../mys_numpy/Validation/train_label.npy")

    # 凝视特征
    gaze_train = np.load("../mys_numpy/Train/gaze_array.npy")
    gaze_validation = np.load("../mys_numpy/Validation/gaze_array.npy")

    # 眼睛坐标
    eye_train = np.load("../mys_numpy/Train/eye_array.npy")
    eye_validation = np.load("../mys_numpy/Validation/eye_array.npy")

    # 头部坐标
    head_train = np.load("../mys_numpy/Train/head_array.npy")
    head_validation = np.load("../mys_numpy/Validation/head_array.npy")

    # 面部坐标
    face_train = np.load("../mys_numpy/Train/face_array.npy")
    face_validation = np.load("../mys_numpy/Validation/face_array.npy")

    # 非刚性面部
    non_ridge_train = np.load("../mys_numpy/Train/non_ridge_array.npy")
    non_ridge_validation = np.load("../mys_numpy/Validation/non_ridge_array.npy")

    # alphapose人体姿势数据
    alphapose_train = np.load("../mys_numpy/Train/num_array.npy")
    alphapose_validation = np.load("../mys_numpy/Validation/num_array.npy")

    # 面部行为单元强度（未标准化）
    itense_train = np.load("../mys_numpy/Train/intense_array.npy")
    itense_validation = np.load("../mys_numpy/Validation/intense_array.npy")

    # 融合特征
    X_train = np.concatenate([gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train], axis=2)
    del gaze_train, eye_train, head_train, face_train, non_ridge_train, itense_train
    X_validation = np.concatenate([gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation], axis=2)
    del gaze_validation, eye_validation, head_validation, face_validation, non_ridge_validation, itense_validation  
    
    Y_train = np_utils.to_categorical(train_label, 4)
    Y_validation = np_utils.to_categorical(validation_label, 4)

    g = Graph(True)
    print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:
      for epoch in range(1, hp.num_epochs):
        if sv.should_stop():
          break
        

    # with sv.managed_session() as sess:
    #     for epoch in range(1, hp.num_epochs+1): 
    #         if sv.should_stop():
    #             break
    #         for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
    #             sess.run(g.train_op)
    #             print(sess.run(g.acc))
    #             gs = sess.run(g.global_step)   
    #         sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
    print("Done")    
    
