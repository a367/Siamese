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

IMG_SIZE = (96, 112)


def make_datum(img, label, channels):
    return caffe_pb2.Datum(
        channels=channels,
        width=IMG_SIZE[0],
        height=IMG_SIZE[1],
        label=label,
        data=img.tostring())


def store_datums(db, imgs, train_num, channels, labels=None):
    for i in xrange(train_num):
        if labels is not None:
            datum = make_datum(imgs[i], labels[i], channels)
        else:
            datum = make_datum(imgs[i], 1, channels)
        db.put('%.8d' % i, datum.SerializeToString())


def create_triplet_lmdb(npz_path, lmdb_name):
    src = np.load(npz_path)

    train_imgs = src['triplet_X']
    train_num = np.shape(train_imgs)[0]

    print train_imgs.shape

    with lmdb.open('data/' + lmdb_name, map_size=int(1e12)).begin(write=True) as db:
        store_datums(db, train_imgs, train_num, 9)


def create_contrastive_lmdb(npz_path, lmdb_name):
    src = np.load(npz_path)

    train_imgs = src['train_X']
    train_sims = src['train_Y']
    train_num = np.shape(train_imgs)[0]

    print train_imgs.shape

    with lmdb.open('data/' + lmdb_name, map_size=int(1e12)).begin(write=True) as db:
        store_datums(db, train_imgs, train_num, 2, train_sims)


def create_test_contrastive_lmdb(npz_path, lmdb_name):
    src = np.load(npz_path)

    train_imgs = src['test_X'].reshape(10000, 2 * 28 * 28)
    train_sims = src['test_Y']
    train_num = np.shape(train_imgs)[0]

    print train_imgs.shape

    with lmdb.open('data/' + lmdb_name, map_size=int(1e12)).begin(write=True) as db:
        store_datums(db, train_imgs, train_num, 2, train_sims)


def create_single_lmdb(npz_path, lmdb_name, data_type):
    src = np.load(npz_path)

    train_imgs = src[data_type + '_X']
    identities = src[data_type + '_Y']
    train_num = np.shape(train_imgs)[0]

    print train_imgs.shape

    with lmdb.open('data/%s_%s' % (data_type, lmdb_name), map_size=int(1e12)).begin(write=True) as db:
        store_datums(db, train_imgs, train_num, 3, labels=identities)


def create_casia_lmdb(npz_path, lmdb_path):
    src = np.load(npz_path)
    train_imgs = src['X']

    num = train_imgs.shape[0]

    with lmdb.open(lmdb_path, map_size=int(1e12)).begin(write=True) as db:
        count = 0
        for i in xrange(num):
            for img in train_imgs[i]:
                datum = make_datum(img, i, 1)
                db.put('%.8d' % count, datum.SerializeToString())
                count += 1


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.nan)
    # create_contrastive_lmdb('data/contrastive_mnist.npz', 'contrastive_lmdb')
    # create_test_contrastive_lmdb('data/contrastive_mnist.npz', 'contrastive_test_lmdb')
    # create_triplet_lmdb('data/triplet_mnist.npz', 'triplet_lmdb')

    # create_triplet_lmdb('data/triplet_lfw.npz', 'triplet_lfw_lmdb')
    # create_single_lmdb('data/lfw.npz', 'lfw_lmdb', data_type='train')
    create_casia_lmdb('data/CASIA/src_casia.npz', 'data/CASIA/casia_lmdb')
