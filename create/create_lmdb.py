# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午8:17
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn


import random
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

# from scipy.misc import toimage
# import matplotlib.pyplot as plt

IMG_SIZE = (28, 28)


def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=3,
        width=IMG_SIZE[0],
        height=IMG_SIZE[1],
        label=label,
        data=img.tostring())


def store_datums(db, imgs, train_num):
    for i in xrange(train_num):
        datum = make_datum(imgs[i], 1)
        db.put('%.8d' % i, datum.SerializeToString())


def main(npz_path):
    src = np.load(npz_path)

    train_imgs = src['train_X']
    train_num = np.shape(train_imgs)[0]

    with lmdb.open('data/train_lmdb', map_size=int(1e12)).begin(write=True) as db:
        store_datums(db, train_imgs, train_num)


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.nan)
    main('data/mnist.npz')
