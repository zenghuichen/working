# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:07:13 2018

# introduction： 这是我为了验证的进行k分类进行验证
# Vector介绍:<channel,hight,width,V,Edge,Class>

@author: 30453
"""
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
os.environ['TF_CPP_LOG_LEVEL']='1'
import sys
from six.moves import urllib
import tarfile
import re
import cv2
from PIL import Image as Image

import scipy.signal as signal
#程序主入口

def maintest():
    im=Image.open(r'./l8.jpg')
    #下面主要开始试验
    #拉普拉斯算子的结果
    laplacianKernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float)
    imarr=np.array(im,dtype=np.float)
    i_con=signal.convolve2d(imarr[:,:,0],laplacianKernel*10,mode='valid',boundary='fill',fillvalue=0)
    plt.imshow(i_con)
# 在原有图片矩阵的基础上完成了相关运算的加载，注意，这里使用tensorflow的方案
def ImgProProcessing(path):
    im=Image.open(path)
    #先进行边缘滤波
    


#将输入的图片解析成相关的向量集 
def getVectList(path):
    image=Image.open(path)
    class Imageshape(object):
        pass
    imgshape=Imageshape()
    result=[]
    imarr=np.array(image)
    if len(imarr.shape)>2:#表示这个单通道的介绍
        h,w,c=imarr.shape
        imgshape.len=len(imarr.shape)
        imgshape.c=c
        imgshape.h=h
        imgshape.w=w
    else:
        h,w=imarr.shape
        imgshape.len=len(imgarr.shape)
        imgshape.c=0
        imgshape.h=h
        imgshape.w=w
    i=0#通道数
    while i<imgshape.c:
        j=0#高度
        while j<imgshape.h:
            t=0#宽度
            while t<imgshape.w:
                pix=0
                VectorT=0
                if imgshape.len>2:
                    pix=imarr[j,t,i]
                    VectorT=np.array([i,j,t,pix,0,0],dtype=np.float)
                else:
                    pix=imarr[j,t]
                    VectorT=np.array([i,j,t,pix,0,0],dtype=np.float)
                result.append(VectorT)
                t=t+1
            j=j+1
        i=i+1
    return result

    