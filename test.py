# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午9:31
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import numpy as np

# a = np.array([[-14.62028503, 385.21429443], [64.88478088, 246.48464966]])
#
# print np.sqrt((a[0, 0] - a[1, 0]) ** 2 + (a[0, 1] - a[1, 1]) ** 2)
# a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int8)

a = np.array([1,2,3,4])
b = np.array([4,3,2,3])


print (a==b).any()