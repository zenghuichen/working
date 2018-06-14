#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:59:30 2018

@author: root
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
# constant variable
#the paremeter for read image and label
data_dir_py=r'E:\intelligentcity\example\tf_car_license_dataset\train_images\train-set'
labelsnametxt='metatxt.txt'
odepth=3
owidth=32
oheight=32
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
#------------read and reshape the image and labels---------------    
def createimgandlabel_train(file,iseval=False):
    if not iseval:
        dicts=unpickle(data_dir_py+'/'+'train')
    else:
        dicts=unpickle(data_dir_py+'/'+'test')
    data=dicts[b'data']
    fine_labels=dicts[b'fine_labels']
    # start create result
    imagelabel=[]
    i=0
    while(i<len(data)):
        temp={}
        temp['image']=data[i]
        temp['labelid']=np.array([fine_labels[i]],dtype=np.uint8)
        imagelabel.append(temp)
        i=i+1
    return imagelabel
def createlabelandname(file):
    metedict=unpickle(data_dir_py+'/'+'meta')
    meta_fine=metedict[b'fine_label_names']
    ltemp={}
    i=0
    while i<len(meta_fine):
        ltemp[i]=meta_fine[i].decode('utf-8')
        i=i+1
    return ltemp


#------ deal with map the id of label and  name of label--------------
def storelabelandname(file,labelsnametxt='metatxt.txt'):
    # note: the method for store the map between id and labelname 
    d=createlabelandname(file)
    s=json.dumps(d)
    meta_path=file+'/'+labelsnametxt
    if tf.gfile.Exists(file):
        tf.gfile.MkDir(file)
    f=open(meta_path,'w')
    f.write(s)
    f.close()
    return meta_path

def loadlabelname(meta_path):  
    # note: the method while make the variable that type is int to be the str 
    #so, name=obj[str(id)]
    f=open(meta_path,'r')
    contents=f.read()
    f.close()
    obj=json.loads(contents)
    return obj
    
#-----------------------generate the file of Image and label----------
def makeexample(image,label):
    example=tf.train.Example(features=tf.train.Features(feature={
        'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }
    ))
    return example
def write_TFRecord(img_lbs,filename):
    if os.path.exists(filename):
        os.remove(filename)
    writer=tf.python_io.TFRecordWriter(filename)
    for temp in img_lbs:
        ex=makeexample(temp['image'].tobytes(),temp['labelid'].tobytes())
        writer.write(ex.SerializeToString())
    writer.close()
    return filename

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

    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float64)

    image = tf.reshape(image, [28, 28, 1])
    label = tf.reshape(label, [10])

    image, label = tf.train.batch([image, label],
            batch_size=16,
            capacity=500)

    return image, label
def main_generateTFRecordset(data_dir_py):
    # path
    trainpath=data_dir_py+'/train.tf'
    evalpath=data_dir_py+'/test.tf'
    labelsnametxt='metatxt.txt'
    # train
    trainils=createimgandlabel_train(data_dir_py,iseval=False)
    write_TFRecord(trainils,trainpath)
    #eval data
    evalils=createimgandlabel_train(data_dir_py,iseval=False)
    write_TFRecord(evalils,evalpath)
    #meta_txt
    metadata=storelabelandname(data_dir_py)
    
def test(x):
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator() # using multi-thread
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) # start queue to read
        return sess.run(x)   