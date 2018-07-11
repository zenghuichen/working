# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:07:13 2018

# introduction： 这是我为了验证的进行k分类进行验证
# Vector介绍:<channel,hight,width,V,Edge,Class>

@author: 30453
"""
#############################################模块调用区########################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#the module of purning model
import csv
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_LOG_LEVEL']='1'  # 错误等级提示
import sys
from six.moves import urllib
import tarfile
import re
import cv2
from PIL import Image as Image

import scipy.signal as signal
sys.stdout.write("环境类库加载完成\n\r")
##############################################################################################################


############################进行滤波方法区，主要是算法前预处理####################################################



#高斯拉普拉斯
def LoG(image,sigma,size,_boundary='symm',_mode='valid'):
    def createLOGKernel(sigma,size):
        H,W=size
        r,c=np.mgrid[0:H:1,0:W:1]
        r=r-(H-1)/2
        c=c-(W-1)/2
        #方差
        sigma2=pow(sigma,2.0)
        norm2=np.power(r,2.0)+np.power(c,2.0)
        LOGKernel=(norm2/sigma2-2)*np.exp(-norm2/(2*sigma2))
        return LOGKernel
    loGKernl=createLOGKernel(sigma,size)
    im_log=signal.convolve2d(image,loGKernl,mode=_mode,boundary=_boundary)
    return im_log
#拉普拉斯算子
def lapalcian(image):
    laplacianKernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float)
    i_con=signal.convolve2d(image,laplacianKernel*10,mode='valid',boundary='fill',fillvalue=0)
    return  i_con

def sobel_v(image):
    laplacianKernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float)
    i_con=signal.convolve2d(image,laplacianKernel*10,mode='valid',boundary='fill',fillvalue=0)
    return  i_con

#膨胀
def ImageDilate():
    #结构元半径
    r=1# 半径结构初始化
    MAX_R=20# 最大的结构元半径
    #展示膨胀效果的窗口
    image=cv2.imread(r'./l8.jpg',cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('sourceImage')
    cv2.imshow('sourceImage',image)
    cv2.namedWindow("dilate",1)
    def nothing(*arg):
        pass
    cv2.createTrackbar("r","dilate",r,MAX_R,nothing)
    while True:
        r=cv2.getTrackbarPos("r","dilate")
        s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
        #膨胀图像
        d=cv2.dilate(image,s)
        i_log1=lapalcian(d)
        plt.imshow(i_log1)
        plt.show()
        #显示膨胀结构
        cv2.imshow('dilate',d)
        ch=cv2.waitKey(5)
        if ch==27:
            break
    cv2.destroyAllWindows()

#腐蚀
def ImageErode():
    I=cv2.imread(r'./l8.jpg',cv2.IMREAD_GRAYSCALE)
    S=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    r=cv2.erode(I,S)
    e=I-r
    cv2.imshow("I",I);
    cv2.imshow("r",r)
    cv2.imshow('e',e)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
##############################################################################################################


#########################################程序主要代码测试区域###################################################

im=Image.open(r'./l8.jpg')# 读取图片
imarr=np.array(im,dtype=np.float)
i_con=lapalcian(imarr[:,:,0])
#plt.imshow(i_con)
#plt.show()
i_log=LoG(imarr[:,:,0],6,(3,3))
#plt.imshow(i_log)
#plt.show()
i_log1=lapalcian(i_log)
#plt.imshow(i_log1)
#plt.show()

#ImageExpand()  #膨胀
ImageErode() #腐蚀

##############################################################################################################
