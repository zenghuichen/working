# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 02:23:11 2018
这个文件主要就是用来完成模型的识别
@author: 30453
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

#专门用来还原的模型结构
import model as cifar10

Eval_dir='/home/chen/文档/mygithub/mycode/cifar100/eval'
log_dir='/home/chen/文档/mygithub/mycode/cifar100/log'

#参数设置
cifar10.BATCH_SIZE=1

def logout(x):
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MkDir(log_dir)
    log_time=time.gmtime()
    log_name='evallog'+str(log_time.tm_year)+str(log_time.tm_mon)+str(log_time.tm_mday)
    f=os.path.join(log_dir,log_name)
    f=open(f,'a')
    f.write('\n'+str(x))
    f.close()

def eval_once(image):
    with tf.Graph().as_default() as g:
        inputimage=tf.placeholder(tf.uint8,shape=[1,cifar10.odepth,cifar10.oheight,cifar10.owidth])
        iimage=tf.transpose(inputimage,[0,2,3,1])
        imaget=tf.cast(iimage,tf.float32)
        print(imaget.shape)
        logit=cifar10.inference(imaget)
        with tf.Session() as sess:
            var_avg = tf.train.ExponentialMovingAverage(cifar10.moving_avg_decay)
            var_to_restore = var_avg.variables_to_restore()
            saver = tf.train.Saver(var_to_restore)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cifar10.Checkoutpointdir)
            if ckpt and ckpt.model_checkpoint_path:  # restore the checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  # 加载模型
            else:
                print('NO checkpoint file found')
                return
            return sess.run(logit,
                            feed_dict={inputimage:image})

def insertTop5(b,i,v):
    if len(b) != 10:
        raise ValueError('数组长度不对，应该为5')
    #进行程序插入
    j=9
    while j >i :
        b[j]=b[j-1]
        j=j-1
    b[i]=v
    return b
def top5(a):
    if len(a)<5:
        raise ValueError('数组长度太少了')
        
    index=0
    b=np.zeros([10],dtype=np.int)
    for at in a:
        j=0
        while j<10:
            if a[b[j]] < at:
                b=insertTop5(b,j,index)
                break
            j=j+1
        index=index+1
    return b


def testimg(path):
    img=Image.open(path)
    imgarr=np.array(img)
    plt.imshow(imgarr)
    imgarr=np.array([imgarr],dtype=np.int)
    imgarr=np.array([imgarr],dtype=np.int)
    d=eval_once(imgarr)
    t=top5(d[0,:])
    print(t)
    return d,t