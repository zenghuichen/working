# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:43:38 2018

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
datalabel=r'E:\intelligentcity\example\tf_car_license_dataset\labelimg.txt'
dataset=r"E:\intelligentcity\example\tf_car_license_dataset\train_images"
datasrc=r'E:\intelligentcity\example\jianceng\02'


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

def parseLabel(meta_path):
    f=open(meta_path,'r')
    contents=f.read()
    f.close()
    metas=contents.replace('\n','').split('},{')
    metalist=[]
    for i in metas:        
        temp=i.replace('{','').replace('}','').split(',')
        meta={}
        print(temp)
        for j in temp:
            print(j)
            tmp=j.split(':')
            meta[tmp[0]]=temp[1]
        metalist.append(meta)
    return metalist

def CreateName(Cls,Num,nn,img):
    d=time.asctime()
    t=str(d).replace(' ','').replace(':','')
    return str(Cls)+'_'+str(Num)+'_'+str(nn)+str(t)+str(img)

def CreateList(filepath):
    #为了梳理
    pass






def logdsss():
    topicsrc=r'E:\intelligentcity\example\tf_car_license_dataset\train_images\train-set'
    tosrc=r'E:\intelligentcity\example\jianceng'
    tslist=[]
    tslist.append({'s':'02','t':42})
    tslist.append({'s':'03','t':43})
    tslist.append({'s':'04','t':44})
    tslist.append({'s':'05','t':45})
    tslist.append({'s':'06','t':46})
    tslist.append({'s':'07','t':48})
    tslist.append({'s':'08','t':52})
    tslist.append({'s':'09','t':54})
    tslist.append({'s':10,'t':55})
    tslist.append({'s':11,'t':56})
    tslist.append({'s':12,'t':58})
    tslist.append({'s':13,'t':60})
    tslist.append({'s':14,'t':61})
    tslist.append({'s':15,'t':67})
    tslist.append({'s':16,'t':68})
    tslist.append({'s':17,'t':75})
    tslist.append({'s':18,'t':71})
    tslist.append({'s':19,'t':62})
    tslist.append({'s':20,'t':63})
    tslist.append({'s':21,'t':64})
    tslist.append({'s':22,'t':65})
    tslist.append({'s':23,'t':66})
    tslist.append({'s':24,'t':72})
    tslist.append({'s':25,'t':73})
    tslist.append({'s':26,'t':47})
    tslist.append({'s':27,'t':57})
    tslist.append({'s':28,'t':76})
    tslist.append({'s':29,'t':74})
    
    for tmp in tslist:
        osrc=os.path.join(tosrc,str(tmp['s']))
        print(tmp)
        dlist=os.listdir(osrc)
        oimglist=list(map(lambda x:os.path.join(osrc,x),dlist))
        j=0
        for i in oimglist:
            #开始完成图片的输出
            img=Image.open(i)
            imgName=CreateName('N',tmp['t'],j,'.bmp')
            img.save(os.path.join(topicsrc,imgName))
            j=j+1
            
topicsrc=r'E:\intelligentcity\example\tf_car_license_dataset\train_images\train-set'
#完成字符解析和大类处理

dlist=os.listdir(topicsrc)
relist=[]
j=0
for tmp in dlist:
    tl=tmp.split('_')
    t={}
    t['s']=tmp
    Cl=int(tl[1])
    if Cl==0:
        t['r']='U'
    elif Cl<11:
        t['r']='N'
    elif Cl<37:
        t['r']='A'
    elif Cl<41:
        t['r']='U'
    elif Cl<75:
        t['r']='C'
    elif Cl<81:
        t['r']='U'
    elif Cl<82:
        t['r']='S'
    else:
        t['r']='U'
    t['r']=CreateName(t['r'],Cl,j,'.bmp')
    j=j+1
    relist.append(t)
for tmp in relist:
    oname=os.path.join(topicsrc,tmp['s'])
    rname=os.path.join(topicsrc,tmp['r'])
    os.rename(oname,rname)

print('over')
