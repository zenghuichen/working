#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:25:24 2018

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
from tensorflow.contrib.model_pruning.python import pruning  #  训练
from tensorflow.python import debug as tf_debug
from functools import reduce # 输出log
import pickle
import json
from PIL import Image
# constant variable
#the paremeter for read image and label
#data_dir='/home/chen/data/data/cifar-100-binary'
labelsnametxt='metatxt.txt'
data_dir=r'E:\intelligentcity\example\tf_car_license_dataset\train_images\train-set'
odepth=1
owidth=32
oheight=40
labelbytes=2# 2 for CIFAR-100
olabel=1
Image_crop_h=40
Image_crop_w=32
NUM_EXAMPLE_TRAIN=11423
NUM_EXAMPLE_EVAL=98
MIN_FRACTION=1
min_queue_example=int(NUM_EXAMPLE_TRAIN*MIN_FRACTION)


#-set the hyperparameter of model

BATCH_SIZE=256
NUM_CLASS=90
NUM_EXAMPLES_PER_EPOCH_FOR_TRIAN=NUM_EXAMPLE_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=NUM_EXAMPLE_EVAL
moving_avg_decay=0.999
learning_rate_init=0.001# 初始学习率
learning_rate_decay_factor=0.0001 #学习率衰减因子
num_epoch_per_decay=350.0 # 衰减成阶梯函数，控制衰减周期（阶梯宽度）
TOWER_NAME='tower'
# in variable
Checkoutpointdir=r'E:\intelligentcity\train'
log_dir=r'E:\home\chen\文档\mygithub\mycode\cifar100\log'

Max_step=1000000
FLASS=None
# visiriable
logstep=1

#-----------------------------generator TfRecordset--------------------------
def unpickle(file): # 用于读取cifar100-python
    with open(file, 'rb') as fo:
         dict = pickle.load(fo, encoding='bytes')
    return dict

#------------------------------input image and labels------------------------
def _createqueue(data_dir):
    filenames=[data_dir]
    for f in filenames:
        if tf.gfile.Exists(f):
            continue
        else:
            print(f)
            raise ValueError('no file need to add the file queue')
    filename_queue=tf.train.string_input_producer(tf.train.match_filenames_once(data_dir+'/*.bmp'))
    return filename_queue

def parsename(k,gc=1):
    k=tf.reshape(k,shape=[1])
    d=tf.string_split(k,delimiter='\\')
    d=d.values[-1]
    d=tf.reshape(d,shape=[1])
    d=tf.string_split(d,delimiter='.')
    d=d.values[0]
    d=tf.reshape(d,shape=[1])
    d=tf.string_split(d,delimiter='_')
    d=d.values[1]
    return d
def read_tfrecord_users(filename_queue):
    class CIFAR100Record(object):
        pass
    
    result=CIFAR100Record()
        
    reader=tf.WholeFileReader()
    k,v = reader.read(filename_queue)
    k=parsename(k)
    k=tf.string_to_number(k,tf.int32)
    k=tf.cast(k,tf.uint8)
    image = tf.image.decode_bmp(v)
    label = tf.reshape(k, [olabel])
    result.image=image
    result.finelabel=label
    return result



def distored_input(data_dir,shuffle=True,Num_process_thread=16):
    filequeue=_createqueue(data_dir)
    read_input=read_tfrecord_users(filequeue)
    fimg=read_input.image
    fimg=tf.cast(fimg,tf.float32) 
    # the size of image croped
    height=Image_crop_h
    width=Image_crop_w
    #  图片处理
       
    fimg=tf.random_crop(fimg,[height,width,odepth])  #图片裁剪
    fimg=tf.image.random_brightness(fimg,63)   # 随机亮度饱和
    fimg=tf.image.per_image_standardization(fimg)  #图片标准化
    fimg.set_shape([height,width,odepth])
    read_input.finelabel.set_shape([1])
    print('gilling queue with %d image ' % min_queue_example)
    if shuffle:
        images,label=tf.train.shuffle_batch(
                [fimg,read_input.finelabel],
                batch_size=BATCH_SIZE,
                num_threads=Num_process_thread,
                capacity=min_queue_example+3*BATCH_SIZE,
                min_after_dequeue=min_queue_example,
                )
    else:
        images,label= tf.train.batch(
                [fimg,read_input.finelabel],
                batch_size=BATCH_SIZE,
                num_threads=Num_process_thread,
                capacity=min_queue_example+3*BATCH_SIZE,
                )
    tf.summary.image('images',images)
    return images,tf.reshape(label,[BATCH_SIZE])

#-----------------------purning model--------------------------  
def _activate_summary(x):
    """
    创建一个直方图,创建一个稀疏矩阵
    :param x:tensor
    :return:nothing
    """
    tensor_name=re.sub('%s_[0-9]*/'%TOWER_NAME,'',x.op.name)  #若多个GPU训练，则从名称中删除，利于tensorboard显示
    tf.summary.histogram(tensor_name+'/activations',x) # 提供激活直方图
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x)) # 提供激活稀疏行

def _variable_on_cpu(name,shape,initializer):
    """

    :param name:变量名
    :param shape: list of ints
    :param initializer: 变量的初始化
    :return:
     variable tensor
    """
    with tf.device('/gpu:0'):
        dtype=tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def _variable_with_weight_decay(name,shape,stddv,wd):

    var=_variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddv,dtype=tf.float32))
    if wd is not None:
        weight_decay=tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('loss',weight_decay)
    return var

def model_image_input(data_dir_py,shuffle=True,Num_process_thread=16):
    """读入图片和标签
    为了获取训练和预测图片输入
    """
    images,labels=distored_input(data_dir_py,shuffle,Num_process_thread=16)
    return images,labels

def inference(images):
    
    # 卷积层1
    with tf.variable_scope('conv1') as scope:  # test 使用 get_variable代替variable() 方法
        kernel=_variable_with_weight_decay('weights',shape=[5,5,odepth,64],stddv=0.05,wd=0.0) #卷积核
        conv=tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')  # 卷积函数
        biases=_variable_on_cpu('biases',[64],tf.constant_initializer(0.0)) # 偏置
        pre_activation=tf.nn.bias_add(conv,biases) # 生成激活函数
        conv1=tf.nn.relu(pre_activation,name=scope.name) # 采用relu激活函数
        _activate_summary(conv1) #汇总
    # 池化层1
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    norm1=tf.nn.lrn(pool1,4,0.1,0.001/9.0,0.75,name='norm1')# 采用lrn激活
    #卷积层2
    with tf.variable_scope('conv2') as scope:
        kernel=_variable_with_weight_decay('weights',shape=[5,5,64,64],stddv=0.05,wd=0.0) #卷积核  注意由于前一层次的输出为 64 个图层
        conv=tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')  # 卷积函数
        biases=_variable_on_cpu('biases',[64],tf.constant_initializer(0.0)) # 偏置
        pre_activation=tf.nn.bias_add(conv,biases) # 生成激活函数
        conv2=tf.nn.relu(pre_activation,name=scope.name) # 采用relu激活函数
        _activate_summary(conv2) #汇总
    norm2=tf.nn.lrn(conv2,4,0.1,0.001/9.0,0.75,name='norm2')
    pool2=tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    #全连接层3
    with tf.variable_scope('local3') as scope:
        reshape=tf.reshape(pool2,[BATCH_SIZE,-1]) #将输入转化为一维向量
        dim=reshape.get_shape()[1].value
        weight=_variable_with_weight_decay('weights',shape=[dim,384],stddv=0.04,wd=0.004) # 使用正则l2 约束防止过拟合
        biases=_variable_on_cpu('biases',[384],initializer=tf.constant_initializer(0.1))
        local3=tf.nn.relu(tf.matmul(reshape,weight)+biases,name=scope.name)
        _activate_summary(local3)
    #全连接层4
    with tf.variable_scope('local4') as scope:
        weights=_variable_with_weight_decay('weights',shape=[384,192],stddv=0.04,wd=0.004)
        biases=_variable_on_cpu('biases',[192],tf.constant_initializer(0.1))
        local4=tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope.name)
        _activate_summary(local4)

    #线性层
    with tf.variable_scope('softmax_linear') as scope:
        weights=_variable_with_weight_decay('weights',[192,NUM_CLASS],stddv=1/192.0,wd=None)
        biases=_variable_on_cpu('biases',[NUM_CLASS],tf.constant_initializer(0.0))
        softmax_linear=tf.add(tf.matmul(local4,weights),biases,name=scope.name)
        _activate_summary(softmax_linear)
    return softmax_linear
# 获得总损失函数
def loss(logits,labels):
    labels=tf.cast(labels,tf.int64) # 方便输入
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    print(cross)
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy') # 计算整个批次的交叉熵
    tf.add_to_collection('losses',cross_entropy_mean) #获取的交叉熵损失  加入到变量集合
    return tf.add_n(tf.get_collection('losses'),name='total_loss')  #   tf.add_n实现列表元素相加
# 添加损失的summary；计算所有单个损失的移动均值和总损失
def _add_loss_summaries(total_loss):
    #滑动模型
    loss_avg=tf.train.ExponentialMovingAverage(0.9,name='avg')# 指数移动平均
    losses=tf.get_collection('losses')
    loss_avg_op=loss_avg.apply(losses+[total_loss]) #将指数移动平均应用于单个损失
    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name+'(raw)', 1)
        tf.summary.scalar(l.op.name, loss_avg.average(l))
    return loss_avg_op

# CIFAR-100模型中的反向传播
def train(total_loss,global_step):
    #影响学习率的变量
    num_batch_pre_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRIAN/BATCH_SIZE  #计算batch数
    decay_step=int(num_batch_pre_epoch*num_epoch_per_decay)  #衰减步数
    #指数衰减学习率
    lr=tf.train.exponential_decay(learning_rate_init,
                                  global_step,
                                  decay_step,
                                  learning_rate_decay_factor,
                                  staircase=True) #计算指数衰减率
    tf.summary.scalar('learning_rate',lr)
    #对总损失进行平均
    loss_avg_op=_add_loss_summaries(total_loss)
    #计算梯度
    with tf.control_dependencies([loss_avg_op]): #指定opt，grads之前一定计算出指数总平均
        opt=tf.train.GradientDescentOptimizer(lr) # 得到梯度计算的执行器
        grads=opt.compute_gradients(total_loss) # 计算梯度
    #应用处理过后的梯度
    apply_grads_op=opt.apply_gradients(grads,global_step=global_step)
    #为可训练变量添加直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)
    #为可梯度添加直方图
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)
    #跟踪所有可训练变量的移动均值
    var_avg=tf.train.ExponentialMovingAverage(moving_avg_decay,global_step)
    var_avg_op=var_avg.apply(tf.trainable_variables())
    #使用默认图形的包装器
    with tf.control_dependencies([apply_grads_op,var_avg_op]):
        train_op=tf.no_op(name='train')
    return train_op
#-----------------------model train-----------------------------
def model_train():
    with tf.Graph().as_default():
        global_step=tf.contrib.framework.get_or_create_global_step() #获取全局变量
        images,labels=model_image_input(data_dir,True,Num_process_thread=16) # 获取图片数据
        logits=inference(images)#生成前向传播模型
        
        losses=loss(logits,labels) #计算损失值

        train_op=train(losses,global_step) #训练模型并更新参数

        pruning_hparams = pruning.get_pruning_hparams().parse(FLASS.pruning_hparams)

        pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

        mask_update_op = pruning_obj.conditional_mask_update_op()

        pruning_obj.add_pruning_summaries()

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self.step=-1
            def before_run(self, run_context):
                self.step=self.step+1
                self.start_time=time.time()
                return tf.train.SessionRunArgs(losses)
            def after_run(self, run_context,  run_values):  #输出训练提示
                duration = time.time() - self.start_time
                loss_value = run_values.results
                sys.stdout.write('\r%d:%f' % (self.step, loss_value))
                if self.step % logstep == 0:
                    num_examples_per_step = 128
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
                    logtexts=format_str % (datetime.datetime.now(), self.step, loss_value, examples_per_sec, sec_per_batch)
                    logout(logtexts)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=Checkoutpointdir,
                hooks=[tf.train.StopAtStepHook(last_step=Max_step),
                       tf.train.NanTensorHook(losses),
                       _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=False,
                                     # device_count = {'CPU' : 0, 'GPU' : 0}
                                      )) as mon_sess:
           # mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess) #专门用来进行debug
        
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                mon_sess.run(mask_update_op)


def init_FLASS():
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--train_dir',
          type=str,
          default=Checkoutpointdir,
          help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
          '--pruning_hparams',
          type=str,
          default='',
          help="""Comma separated list of pruning-related hyperparameters""")
    parser.add_argument(
          '--max_steps',
          type=int,
          default=1000000,
          help='Number of batches to run.')
    parser.add_argument(
          '--log_device_placement',
          type=bool,
          default=False,
          help='Whether to log device placement.')
    
    FLASS, unparsed = parser.parse_known_args()
    print('FLASS is initializer sucessfully')
    return FLASS, unparsed 
FLASS,unparsed = init_FLASS()

#----------------------model evals-----------------------------


#---------------------train start----------------------
def main(argv=None):
    FLASS, unparsed =init_FLASS()
    if tf.gfile.Exists(FLASS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLASS.train_dir)
    model_train()

#---------------------system test---------------------------------
def temp():
    pass
def getTimeName():
    d=time.asctime()
    t=str(d).replace(' ','').replace(':','')
    return t
   
def logout(x):
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MkDir(log_dir)
    log_time=time.gmtime()
    log_name='log'+str(log_time.tm_year)+str(log_time.tm_mon)+str(log_time.tm_mday)
    f=os.path.join(log_dir,log_name)
    f=open(f,'a')
    f.write('\n'+str(x))
    f.close()

def testimgs(x,images,labels):
    datalist=[]
    i=0
    print('\n')
    while i<len(labels):
        if labels[i]==x:
            datalist.append(i)
            print(labels[i])
        i=i+1
    print(datalist)
    if len(datalist)==0:
        return
    for t in datalist:
        print(t)
        plt.imshow(images[t,:,:,:])
        plt.show()
        

def test(x):
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator() # using multi-thread
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) # start queue to read
        return sess.run(x)     
#------------------------------脚本程序的运行入口-----------------------------
print('运行前，请检查参数列表的值是不是正确')
#main()
def model_train_main():
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
   pass
#--------generator test and train dataset--------------------------------

 

