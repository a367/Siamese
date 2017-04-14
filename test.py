# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午9:31
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import os
import lmdb
import random
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import matplotlib.pyplot as plt
from scipy.misc import toimage

# a = np.array([[-14.62028503, 385.21429443], [64.88478088, 246.48464966]])
#
# print np.sqrt((a[0, 0] - a[1, 0]) ** 2 + (a[0, 1] - a[1, 1]) ** 2)
# a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int8)

# a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
# b = np.array([[3, 2, 1], [6, 5, 4]])
#
# print np.sum(np.square(a), axis=1)
#
# a = a.astype(np.float) / np.sum(np.square(a), axis=1).reshape(-1, 1)
#
# print a

# print b.T * a

# a = np.array([1,2,3]).reshape(1,-1)
#
# print a.repeat(3,axis=0)

# b = np.array([1,0,1])
#
# c = np.array([0,1,1])
#
# print a*b.reshape(-1,1) * c.reshape(-1,1)

# a = np.array([1,0,1,0]).reshape(-1,1)
# b = np.array([1,2,3,4])
#
# print a*b

# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# b = np.array([[[9,10],[11,12]]])
#
# print np.vstack((a,b))
#
# # b = np.array([[[11,11]],[[11,11]]])
# #
# # print a.shape
# # print b.shape
# #
# # print np.hstack((a,b))
# print eval('a.shape')
