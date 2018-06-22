# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:28:02 2018
本文档的主要目的是为了二值化。
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
