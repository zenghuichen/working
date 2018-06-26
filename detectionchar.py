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


def logout(x):
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MkDir(log_dir)
    log_time=time.gmtime()
    log_name='evallog'+str(log_time.tm_year)+str(log_time.tm_mon)+str(log_time.tm_mday)
    f=os.path.join(log_dir,log_name)
    f=open(f,'a')
    f.write('\n'+str(x))
    f.close()

def eval_once(image,num):
    cifar10.BATCH_SIZE=num
    with tf.Graph().as_default() as g:
        inputimage=tf.placeholder(tf.uint8,shape=[num,cifar10.odepth,cifar10.oheight,cifar10.owidth])
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
            t= sess.run(logit,feed_dict={inputimage:image})
<<<<<<< HEAD
            oplist=[]
            for var in tf.trainable_variables():
                print(var.name)
                oplist.append(var)
            # start out the result of lasyer
            kernel=oplist[0]
            biases=oplist[1]
            conv=tf.nn.conv2d(imaget,kernel,[1,1,1,1],padding='SAME')
            pre_activation=tf.nn.bias_add(conv,biases)
            conv1=tf.nn.relu(pre_activation,name="conv1tmp")
            print(conv1)
            convsess=sess.run(conv1,feed_dict={inputimage:image})
            return t,convsess
=======
            for var in tf.trainable_variables():
                print(var.name)
            return t
>>>>>>> 1033ca4532641b7f8e087332e53f4a56d5059a7e

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
    num=len(imglist)
    img=Image.open(path)
    img=img.resize([cifar10.oheight,cifar10.owidth])
    imgarr=np.array(img)
    print(imgarr.shape)
    imgarr=np.array([imgarr],dtype=np.int)
    imgarr=np.array([imgarr],dtype=np.int)
    d=eval_once(imgarr,num)
    t=top5(d[0,:])
    plt.imshow(img)
    plt.show()
   # print(t)
    tm=list(map(mapLabel,t))
    print(tm[0])
    return d,t,tm



def mapLabel(x):
    mp={0:'None',11:'A',21:'K',31:'U',41:'京',51:'浙',61:'琼',71:'藏',81:'分割符',
        1:'0',12:'B',22:'L',32:'V',42:'津',52:'皖',62:'陕',72:'港',82:'',
        2:'1',13:'C',23:'M',33:'W',43:'冀',53:'闽',63:'甘',73:'澳',83:'',
        3:'2',14:'D',24:'N',34:'X',44:'晋',54:'赣',64:'青',74:'台',84:'',
        4:'3',15:'E',25:'O',35:'Y',45:'蒙',55:'鲁',65:'宁',75:'云',85:'',
        5:'4',16:'F',26:'P',36:'Z',46:'辽',56:'豫',66:'新',76:'贵' ,86:'',
        6:'5',17:'G',27:'Q',37:' ',47:'吉',57:'鄂',67:'渝',77:' ' ,87:'',
        7:'6',18:'H',28:'R',38:' ',48:'黑',58:'湘',68:'川',78:' ' ,88:'',
        8:'7',19:'I',29:'S',39:' ',49:'沪',59:'粤',69:'黔',79:' ' ,89:'',
        9:'8',20:'J',30:'T',40:' ',50:'苏',60:'桂',70:'滇',80:' ' ,90:'',
        10:'9'}
    result=mp[x]
    return result.replace(' ','')



path=r'/home/chen/working/data/example/tf_car_license_dataset/test_images'
imgs=os.listdir(path)
pathlist=list(map(lambda x:os.path.join(path,x),imgs))
imglist=[]
i=pathlist[0]
img=Image.open(i)
img=img.resize([cifar10.owidth,cifar10.oheight])
imgarr=np.array(img)
imgarr=np.array([imgarr],dtype=np.int)
imglist.append(imgarr)
imgs=np.array(imglist)
num=len(imglist)
<<<<<<< HEAD
d,conv1v=eval_once(imgs,num)
=======
d=eval_once(imgs,num)
>>>>>>> 1033ca4532641b7f8e087332e53f4a56d5059a7e
# data map to label
if d.shape[0]==imgs.shape[0]:
    i=0
    while i<imgs.shape[0]:
        li=d[i,:]
        imgi=imgs[i,0,:,:]
        t=top5(li)
<<<<<<< HEAD
       # print(t)
        tm=list(map(mapLabel,t))
        print(tm[0])
=======
        print(t)
        tm=list(map(mapLabel,t))
        print(tm)
>>>>>>> 1033ca4532641b7f8e087332e53f4a56d5059a7e
        plt.imshow(imgi)
        plt.show()
        i=i+1
        