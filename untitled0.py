# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:05:35 2018

@author: 30453
"""

#the module of input
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#the module of purning model
import csv
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_LOG_LEVEL']='2'
import sys
from six.moves import urllib
import tarfile
import re
#model train
import argparse
import time
import datetime
import pickle
import json
from tensorflow.contrib.model_pruning.python import pruning  #  训练
from functools import reduce # 输出log
from PIL import Image
import numpy as np

data_dir_py=r'E:\intelligentcity\example\tf_car_license_dataset\train_images\train-set'
tddir=r'E:\intelligentcity\example\tfrecord\train.tf'
filename=tddir
#-------------------local variable-------------------------------------------
width=32
height=40
channels=1

#--------------------test  uint-----------------------------------------------
def test(x):
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator() # using multi-thread
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) # start queue to read
        return sess.run(x)   


#--------------------log uint--------------------------------------------------

def unpickle(file): # 用于读取cifar100-python
    with open(file, 'rb') as fo:
         dict = pickle.load(fo, encoding='bytes')
    return dict

def logout(x):
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MkDir(log_dir)
    log_time=time.gmtime()
    log_name='log'+str(log_time.tm_year)+str(log_time.tm_mon)+str(log_time.tm_mday)
    f=os.path.join(log_dir,log_name)
    f=open(f,'a')
    f.write('\n'+str(x))
    f.close()  
    
#---------------------------------read origin image-----------------------------

def mapImg(logpath):
    imglist=os.listdir(logpath)
    result=[]
    for tmp in imglist:
        name=tmp.split('_')
        label=name[1]
        mate={}
        mate['name']=os.path.join(logpath,tmp)
        mate['label']=int(label)
        result.append(mate)
    return result
def readImage(tmp):
    img=Image.open(tmp['name'])
    img=img.resize(size=[width,height])
    label=int(tmp['label'])
    #zip the dimension of img
    imgarr=np.array(img,dtype=np.uint8)
    h,w=imgarr.shape
    imgtmp=np.zeros(h*w,dtype=np.uint8)
    i=0
    while i<h:
        j=0
        while j<w:
            imgtmp[w*i+j]=imgarr[i,j]
            j=j+1
        i=i+1
    result={}
    result['image']=imgtmp
    result['labelid']=np.array([1],dtype=np.uint)
    result['labelid'][0]=int(label)
    return result

    
#--------------------------------create TFrecord--------------------------------
def makeexample(image,label):
    example=tf.train.Example(features=tf.train.Features(feature={
        'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }
    ))
    return example
def write_TFRecord(logpath,filename):
    if os.path.exists(filename):
        os.remove(filename)
    writer=tf.python_io.TFRecordWriter(filename)
    rlist=mapImg(logpath)
    i=0
    print('start')
    for tmp in rlist:
        temp=readImage(tmp)
        ex=makeexample(temp['image'].tobytes(),temp['labelid'].tobytes())
        writer.write(ex.SerializeToString())
        print('\r'+str(i)+'',end='')
        i=i+1
    print('')
    writer.close()
    return filename

#write_TFRecord(data_dir_py,tddir)

#-----------------------------------read image --------------------------------

def read_tfrecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)

    image = tf.reshape(image, [height, width,channels])
    label = tf.reshape(label, [1])

    image, label = tf.train.batch([image, label],
            batch_size=32,
            capacity=900)

    return image, label    