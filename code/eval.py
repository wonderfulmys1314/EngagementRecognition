# -*- coding: utf-8 -*-
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

# from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from train import Graph
from sklearn.preprocessing import OneHotEncoder


enc = OneHotEncoder()

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## 重载数目
            test_label = np.load("../mys_numpy/Test/train_label.npy")

            # 凝视特征
            gaze_test = np.load("../mys_numpy/Test/gaze_array.npy")

            # 眼睛坐标
            eye_test = np.load("../mys_numpy/Test/eye_array.npy")

            # 头部坐标
            head_test = np.load("../mys_numpy/Test/head_array.npy")

            # 面部坐标
            face_test = np.load("../mys_numpy/Test/face_array.npy")

            # 非刚性面部
            non_ridge_test = np.load("../mys_numpy/Test/non_ridge_array.npy")

            # alphapose人体姿势数据
            alphapose_test = np.load("../mys_numpy/Test/num_array.npy")


            # 面部行为单元强度（未标准化）
            itense_test = np.load("../mys_numpy/Test/intense_array.npy")

            # 融合特征
            X = np.concatenate([gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test], axis=2)
            del gaze_test, eye_test, head_test, face_test, non_ridge_test, itense_test
            Y = np_utils.to_categorical(test_label, 4)

            # 合并数据集
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## 得到模型名字
            # mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

            y_Pred, y_True, test_acc = sess.run([g.preds, g.y_true, g.acc], {g.x:X, g.y:Y})

            print(test_acc)
            print(y_Pred)
            print(y_True)
            
                                          
if __name__ == '__main__':
    eval()
    print("Done")

    
    