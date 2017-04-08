# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午7:43
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn


import os
import lmdb
import numpy as np
from PIL import Image
import caffe
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import matplotlib.pyplot as plt
from scipy.misc import toimage
from google.protobuf import text_format
from rms import SLDataSet, train_msrl


def generate_mean_image():
    os.system('~/caffe/build/tools/compute_image_mean -backend=lmdb data/train_lmdb data/mean.binaryproto')


def get_data_for_case_from_lmdb(lmdb_name, idx):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()

    raw_datum = lmdb_txn.get(idx)
    datum = caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    feature = datum_to_array(datum)
    label = datum.label

    return label, feature


def train(solver_path, log_path, gpu):
    os.system('~/caffe/build/tools/caffe train --solver %s -gpu %s  2>&1 | tee %s' % (solver_path, gpu, log_path))


def compute_accuracy(net, M, cnn_test_X_1, cnn_test_X_2, sims, threshold, use_rms):
    # print cnn_test_X_1
    # print cnn_test_X_2


    if use_rms:
        # dis = np.diag(np.dot(np.dot(cnn_test_X_1.T, M), cnn_test_X_2))
        dis = np.sum(np.dot(cnn_test_X_1.T, M).T * cnn_test_X_2, axis=0)
    else:
        dis = np.sqrt(np.sum((cnn_test_X_1 - cnn_test_X_2) ** 2, axis=0))
    normal_dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))

    if use_rms:
        pred = normal_dis >= threshold
    else:
        pred = normal_dis <= threshold

    # confusion matrix
    TP = np.sum(pred * sims)
    FP = np.sum(pred * (1 - sims))
    FN = np.sum((1 - pred) * sims)
    TN = np.sum((1 - pred) * (1 - sims))

    accuracy = np.mean(pred == sims)
    TPR = TP / float(TP + FN)
    FPR = FP / float(TN + FP)

    return accuracy, TPR, FPR


def apply(deploy_path, npz_path, caffemodel_path, params, use_rms=True, use_l2=False):
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_path, caffemodel_path, caffe.TEST)

    # feature type
    feat_type = "l2_feat" if use_l2 else "feat"

    # load data
    data = np.load(npz_path)

    ##############
    # train data #
    ##############
    train_X = data['train_X'].reshape(60000, 3, 28, 28)

    src_train_X_a = train_X[:, :1, :, :].astype(np.uint8)
    src_train_X_p = train_X[:, 1:2, :, :].astype(np.uint8)
    src_train_X_m = train_X[:, 2:, :, :].astype(np.uint8)
    print 'train_X shape:', train_X.shape

    net.blobs['data'].reshape(*src_train_X_a.shape)
    cnn_train_X_a = net.forward(data=src_train_X_a)[feat_type].copy()
    cnn_train_X_p = net.forward(data=src_train_X_p)[feat_type].copy()
    cnn_train_X_m = net.forward(data=src_train_X_m)[feat_type].copy()

    ##############
    # test data  #
    ##############
    test_X = data['test_X']
    test_Y = data['test_Y']

    imgs_1 = test_X[:, :1, :, :].astype(np.uint8)
    imgs_2 = test_X[:, 1:, :, :].astype(np.uint8)
    print 'test_X shape:', test_X.shape

    net.blobs['data'].reshape(*imgs_1.shape)
    cnn_test_X_1 = net.forward(data=imgs_1)[feat_type].copy().T.astype("float32")
    cnn_test_X_2 = net.forward(data=imgs_2)[feat_type].copy().T.astype("float32")

    print 'train cnn is ok!'

    ##############
    # test phase #
    ##############
    M = None
    if use_rms:
        train_data_set = SLDataSet({'train_X': cnn_train_X_a, 'train_X_plus': cnn_train_X_p, 'train_X_minus': cnn_train_X_m})
        M = train_msrl(train_data_set, params)
    print 'M:'
    print M

    TPR_arr = []
    FPR_arr = []

    point_num = 50
    for i in xrange(point_num):
        accuracy, TPR, FPR = compute_accuracy(net, M, cnn_test_X_1, cnn_test_X_2, test_Y, i / float(point_num), use_rms)
        print 'accuracy: %s,   TPR: %s,   FPR: %s' % (accuracy, TPR, FPR)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)

    print 'TPR =', TPR_arr
    print 'FPR =', FPR_arr


def create_solver_prototxt(src_solover_path, prefix):
    # get source solver config
    solver_config = caffe_pb2.SolverParameter()
    with open(src_solover_path, 'r') as f:
        text_format.Merge(f.read(), solver_config)

    # change some parameters
    solver_config.snapshot_prefix = prefix

    # set temp new config
    new_solver_config = text_format.MessageToString(solver_config)
    with open('tmp/temp.prototxt', 'w') as f:
        f.write(new_solver_config)


def run_facenet(train_name, gpu):
    os.system('rm -rf log/face/%s; mkdir log/face/%s' % (train_name, train_name))
    os.system('mkdir log/face/%s/snapshot' % train_name)

    create_solver_prototxt('model/face/face_solver.prototxt', "log/face/%s/snapshot/face_train" % train_name)
    train('tmp/temp.prototxt', 'log/face/%s/train.log' % train_name, gpu=gpu)


def run_triplet_mnist(train_name, gpu):
    os.system('rm -rf log/triplet_mnist/%s; mkdir log/triplet_mnist/%s' % (train_name, train_name))
    os.system('mkdir log/triplet_mnist/%s/snapshot' % train_name)

    create_solver_prototxt('model/triplet_mnist/triplet_solver.prototxt', "log/triplet_mnist/%s/snapshot/siamese_train" % train_name)
    train('tmp/temp.prototxt', 'log/triplet_mnist/%s/train.log' % train_name, gpu=gpu)


def main():
    # train triplet loss
    # train('model/triplet_solver.prototxt', 'log/train.log')

    # train contrastive loss
    # train('model/siamese/siamese_solver.prototxt', 'log/siamese/train.log')

    # params = {'lr': 1e-1, 'ep': 200, 'lambda': 0, 'epsilon': 10e0, 'mu': 0.5}
    # deploy_path = "model/triplet_l2_deploy.prototxt"
    # data_path = "data/triplet_mnist.npz"
    # caffemodel_path = "log/triplet_l2_50k.caffemodel"
    #
    # apply(deploy_path, data_path, caffemodel_path, params, use_rms=True, use_l2=True)

    # run_triplet_mnist('l2_50k_hard_neg_loss2', 2)
    run_facenet('l2_50k_hard_neg_loss2', 3)

if __name__ == '__main__':
    main()
