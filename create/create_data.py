# -*- coding: utf-8 -*-
# @Time    : 2017/2/12 上午9:35
# @Author  : Zhixin Piao
# @Email   : piaozhx@seu.edu.cn

from PIL import Image, ImageDraw
import numpy as np
import scipy.io as sio
import struct
import random
import json
import os

DATA_SET_PATH = '../data'


class DataSet:
    def __init__(self, npz_path):
        res = np.load(npz_path)

        self.train_X = res['train_X']
        self.train_Y = res['train_Y']
        self.test_X = res['test_X']
        self.test_Y = res['test_Y']

        self.num_examples, self.x_feature_size = np.shape(self.train_X)
        self.y_feature_size = np.shape(self.train_Y)[1]

    def get_batch(self, index, batch_size):
        batch_xs = self.train_X[batch_size * index:batch_size * (index + 1), :]
        batch_ys = self.train_Y[batch_size * index:batch_size * (index + 1), :]

        return batch_xs, batch_ys

    def get_data_shape(self):
        return self.x_feature_size, self.y_feature_size


def str_x2arr(x):
    arr_x = [.0] * 357

    for data in x:
        colon_idx = data.find(':')
        idx = int(data[:colon_idx]) - 1
        dest_data = float(data[colon_idx + 1:])

        arr_x[idx] = dest_data

    return np.array(arr_x)


def str_y2arr(y):
    arr_y = [.0] * 3
    arr_y[int(y)] = 1.0

    return np.array(arr_y)


def create_data():
    X = []
    Y = []

    for line in open('data_set/protein/protein', 'r'):
        vec = line.split()
        x = line.split()[1:]

        for i in xrange(len(x)):
            if x[i][-1] == ':':
                x[i] += x[i + 1]

        str_x = filter(lambda x: x.find(':') != -1, x)
        str_y = vec[0]

        arr_x = str_x2arr(str_x)
        arr_y = str_y2arr(str_y)

        X.append(arr_x)
        Y.append(arr_y)

    X = np.vstack(X)
    Y = np.vstack(Y)

    num_examples = np.shape(X)[0]
    train_num = int(num_examples * 0.7)
    test_num = num_examples - train_num

    total_idxs = range(num_examples)
    train_idxs = random.sample(total_idxs, train_num)
    test_idxs = filter(lambda x: x not in train_idxs, total_idxs)

    train_X = X[train_idxs, :]
    train_Y = Y[train_idxs, :]
    test_X = X[test_idxs, :]
    test_Y = Y[test_idxs, :]

    np.savez('data_set/protein/protein.npz', train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def read_mnist_image(file_path):
    with open(file_path, 'rb') as f:
        s = f.read(16)
        magic_num, img_num, row_num, col_num = struct.unpack(">iiii", s)
        print magic_num

        s = f.read()
        pixel_data = struct.unpack(">" + "B" * img_num * row_num * col_num, s)

        res = []
        for n in xrange(img_num):
            img = [pixel_data[row_num * col_num * n + i] / 255.0 for i in xrange(row_num * col_num)]
            res.append(img)

    return np.array(res)


def read_mnist_label(file_path):
    with open(file_path, 'rb') as f:
        s = f.read(8)
        magic_num, item_num = struct.unpack(">ii", s)
        print magic_num

        s = f.read()
        item_data = struct.unpack(">" + "B" * item_num, s)

        res = []
        for n in xrange(item_num):
            num = item_data[n]
            y = [.0] * 10
            y[num] = 1.0
            res.append(y)

    return np.array(res)


def create_mnist_data():
    train_X = read_mnist_image('data_set/MNIST_data/train-images-idx3-ubyte')
    train_Y = read_mnist_label('data_set/MNIST_data/train-labels-idx1-ubyte')
    test_X = read_mnist_image('data_set/MNIST_data/t10k-images-idx3-ubyte')
    test_Y = read_mnist_label('data_set/MNIST_data/t10k-labels-idx1-ubyte')

    np.savez('data_set/MNIST_data/mnist.npz', train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def get_triple_data(res_num, src_num, class_num, X, Y):
    class2train_x = [[] for i in xrange(class_num)]

    for i in xrange(src_num):
        for j, y in enumerate(Y[i]):
            if y != 0.0:
                class2train_x[j].append(X[i])

    class2train_num = [len(class2train_x[i]) for i in xrange(class_num)]

    X = []
    X_plus = []
    X_minus = []

    for _ in xrange(res_num):
        x_class, x_minus_class = random.sample(range(class_num), 2)

        x_plus_idx = x_idx = random.randint(0, class2train_num[x_class] - 1)
        x_minus_idx = random.randint(0, class2train_num[x_minus_class] - 1)
        while x_plus_idx == x_idx:
            x_plus_idx = random.randint(0, class2train_num[x_class] - 1)

        X.append(class2train_x[x_class][x_idx])
        X_plus.append(class2train_x[x_class][x_plus_idx])
        X_minus.append(class2train_x[x_minus_class][x_minus_idx])

    return np.array(X).T, np.array(X_plus).T, np.array(X_minus).T


def create_mnist_triple_data(npz_path):
    src = np.load(npz_path)

    # source data
    src_train_X = src['train_X']
    src_train_Y = src['train_Y']
    src_test_X = src['test_X']
    src_test_Y = src['test_Y']

    # source parameter
    src_train_num = np.shape(src_train_X)[0]
    src_test_num = np.shape(src_test_X)[0]
    class_num = np.shape(src_train_Y)[1]

    # triple data parameter
    train_num = 70000
    test_num = 30000

    # create triple data
    train_X, train_X_plus, train_X_minus = get_triple_data(train_num, src_train_num, class_num, src_train_X, src_train_Y)
    test_X, test_X_plus, test_X_minus = get_triple_data(test_num, src_test_num, class_num, src_test_X, src_test_Y)

    np.savez('data_set/triple_mnist.npz', train_X=train_X, train_X_plus=train_X_plus, train_X_minus=train_X_minus,
             test_X=test_X, test_X_plus=test_X_plus, test_X_minus=test_X_minus)


def create_CASIA_data(imgs_path, shape=(96, 112)):
    count = 0
    X = []
    for i, (parent, dirnames, filenames) in enumerate(os.walk(imgs_path)):
        if i == 0:
            continue
        identity = []
        for img_name in filenames:
            if img_name == 'Thumbs.db':
                continue
            img_path = os.path.join(parent, img_name)
            img = Image.open(img_path)
            img = img.resize(shape)
            img = np.array(img).astype(np.uint8)

            # change img data structure to (width, height)
            img = np.rollaxis(img, 1)
            identity.append(img.reshape(1, 96, 112))

        X.append(identity)

    X = np.array(X)
    print X.shape
    np.savez('data/CASIA/src_casia.npz', X=X)


def create_single_CASIA_data(src_npz_path, shape=(1, 96, 112)):
    src = np.load(src_npz_path)
    train_imgs = src['X']

    id_num = train_imgs.shape[0]
    input_data = []
    input_label = []
    for i in xrange(id_num):
        for img_data in train_imgs[i]:
            input_data.append(np.array(img_data).reshape(shape))
            input_label.append(i)

    input_data = np.array(input_data)
    input_label = np.array(input_label)

    input_num = len(input_data)
    sample_idx = range(input_num)
    random.shuffle(sample_idx)

    input_data = input_data[sample_idx]
    input_label = input_label[sample_idx]

    print 'train_X shape:', input_data.shape
    print 'train_Y shape:', input_label.shape

    np.savez('data/CASIA/casia.npz', train_X=input_data, train_Y=input_label)


def create_src_mnist_data(npz_path, shape=(1, 28, 28)):
    src = np.load(npz_path)
    train_X = src['train_X']
    train_Y = src['train_Y']

    train_num = train_X.shape[0]

    X = [[] for _ in xrange(10)]
    for i in xrange(train_num):
        for j in xrange(10):
            if train_Y[i][j] != 0:
                X[j].append(train_X[i].reshape(shape).astype(np.uint8))

    np.savez('data/mnist/src_mnist.npz', X=X)


def create_test_mnist_data(npz_path, shape=(1, 28, 28)):
    src = np.load(npz_path)
    test_X = src['test_X']
    test_Y = src['test_Y']

    test_num = test_X.shape[0]

    X = [[] for _ in xrange(10)]
    for i in xrange(test_num):
        for j in xrange(10):
            if test_Y[i][j] != 0:
                X[j].append(test_X[i].reshape(shape).astype(np.uint8))

    contrastive_num = 10000
    match_num = dismatch_num = contrastive_num / 2

    test_X1 = []
    test_X2 = []
    test_Y = []
    for i in xrange(match_num):
        cid = random.randint(0, 9)
        img1, img2 = random.sample(X[cid], 2)

        test_X1.append(img1)
        test_X2.append(img2)
        test_Y.append(1)

    for i in xrange(dismatch_num):
        cid1, cid2 = random.sample(range(10), 2)

        test_X1.append(random.choice(X[cid1]))
        test_X2.append(random.choice(X[cid2]))
        test_Y.append(0)

    test_X1 = np.array(test_X1)
    test_X2 = np.array(test_X2)
    test_Y = np.array(test_Y)

    print 'test_X1 shape:', test_X1.shape
    print 'test_X2 shape:', test_X2.shape
    print 'test_Y shape:', test_Y.shape

    np.savez('data/mnist/test_mnist.npz', test_X1=test_X1, test_X2=test_X2, test_Y=test_Y)


def create_test_lfw_data(npz_path, shape=(1, 96, 112)):
    def get_grey_img(array_img):
        src_img = np.rollaxis(array_img, 2)
        src_img = np.rollaxis(src_img, 1, 3)

        ret_img = np.array(Image.fromarray(src_img).convert('L'))
        ret_img = np.rollaxis(ret_img, 1).reshape(shape)

        return ret_img

    src = np.load(npz_path)
    X = src['X']
    Y = src['Y']

    pair_num = 600
    fold_num = 10

    test_X1 = [[] for _ in xrange(fold_num)]
    test_X2 = [[] for _ in xrange(fold_num)]
    test_Y = [[] for _ in xrange(fold_num)]

    for i in xrange(fold_num):
        for j in xrange(pair_num):
            img1 = get_grey_img(X[i * pair_num + j][0])
            img2 = get_grey_img(X[i * pair_num + j][1])

            test_X1[i].append(img1)
            test_X2[i].append(img2)
            test_Y[i].append(Y[i * pair_num + j])
            # img.save('test.jpg')

    test_X1 = np.array(test_X1)
    test_X2 = np.array(test_X2)
    test_Y = np.array(test_Y)

    np.savez('data/lfw/test_lfw.npz', test_X1=test_X1, test_X2=test_X2, test_Y=test_Y)


def create_person_reid_train_test_data(mat_path):
    data = sio.loadmat(mat_path)

    x_view1 = data['Xview1'].T
    x_view2 = data['Xview2'].T
    x_view = np.array([x_view1, x_view2])

    total_num = x_view1.shape[0]
    train_num = total_num / 2
    test_num = total_num - train_num
    random_idx = range(total_num)

    src_train_X = []
    src_train_Y = []
    src_test_X = []
    src_test_Y = []
    triplet_train_X = []
    test_X = []
    test_Y = []

    for _ in xrange(10):
        random.shuffle(random_idx)

        # divide train_view and test_view
        random_x_view = x_view[:, random_idx, :]
        train_view = random_x_view[:, :train_num, :]
        test_view = random_x_view[:, train_num:, :]

        # add source train/test_view in source train/test data set
        src_train_X.append(train_view)
        src_train_Y.append([1] * train_num)
        src_test_X.append(test_view)
        src_test_Y.append([1] * test_num)

        # create negative case of train view
        neg_train_view = [[]]
        for i in xrange(train_num):
            id1 = i
            while id1 == i:
                id1 = random.randint(0, train_num - 1)
            neg_train_view[0].append(train_view[random.randint(0, 1), id1])

        # set negative case in triplet train view:
        neg_train_view = np.array(neg_train_view)
        train_view = np.vstack((train_view, neg_train_view))
        triplet_train_X.append(train_view)

        # create negative case of test view
        neg_test_view = [[], []]
        sample_ids = range(test_num)
        for i in xrange(test_num):
            id1, id2 = random.sample(sample_ids, 2)
            neg_test_view[0].append(test_view[0, id1, :])
            neg_test_view[1].append(test_view[1, id2, :])

        # set negative case in test view
        neg_test_view = np.array(neg_test_view)
        test_view = np.hstack((test_view, neg_test_view))

        # shuffle test view data
        random_test_idx = range(test_num * 2)
        random.shuffle(random_test_idx)
        test_view = test_view[:, random_test_idx, :]
        test_label = np.array([1] * test_num + [0] * test_num)
        test_label = test_label[random_test_idx]

        # add source train_view and test_view in source train data set and test data set
        test_X.append(test_view)
        test_Y.append(test_label)

    # change type to np.array and output data shape
    data_list = ['src_train_X', 'src_train_Y', 'src_test_X', 'src_test_Y', 'triplet_train_X', 'test_X', 'test_Y']
    for data_name in data_list:
        exec ('%s = np.array(%s)' % (data_name, data_name))
        print '%s shape: %s' % (data_name, eval('%s.shape' % data_name))

    np.savez('data/person-reid/viper.npz', **dict([(n, eval(n)) for n in data_list]))


def compute_accuracy_by_dis(dis, sims, threshold):
    normal_dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))
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


def create_person_reid_triplet_train_test_data(npz_path):
    src = np.load(npz_path)

    train_X1 = src['src_train_X'][0, 0, :, :]
    train_X2 = src['src_train_X'][0, 1, :, :]
    test_X = src['src_test_X'][0]
    test_Y = src['src_test_Y'][0]

    num = train_X1.shape[0]

    triplet_X = [[], [], []]

    for i in xrange(num):
        triplet_X[0] += [train_X1[i].copy() for _ in xrange((num - 1) * 2)]
        triplet_X[1] += [train_X2[i].copy() for _ in xrange((num - 1) * 2)]
        for j in xrange(num):
            if j == i:
                continue
            triplet_X[2].append(train_X1[j])
            triplet_X[2].append(train_X2[j])

    triplet_X = np.array(triplet_X)
    triplet_num = triplet_X.shape[1]

    # shuffle triplet_X
    random_idx = range(triplet_num)
    random.shuffle(random_idx)
    triplet_X = triplet_X[:, random_idx, :]

    print 'triplet_X shape:', triplet_X.shape
    print 'test_X shape:', test_X.shape
    print 'test_Y shape:', test_Y.shape

    np.savez('data/person-reid/triplet_viper.npz', triplet_X=triplet_X, test_X=test_X, test_Y=test_Y)


def create_person_reid_small_triplet_data(l1_norm=True):
    src = np.load('data/person-reid/triplet_viper.npz')
    triplet_X = src['triplet_X']
    num = triplet_X.shape[1]
    triplet_X = triplet_X[:, :num / 10, :]

    train_a = triplet_X[0, :, :]
    train_p = triplet_X[1, :, :]
    train_n = triplet_X[2, :, :]
    test_X1 = src['test_X'][0, :, :]
    test_X2 = src['test_X'][1, :, :]

    data_list = ['train_a', 'train_p', 'train_n', 'test_X1', 'test_X2']

    for data_name in data_list:
        if l1_norm:
            exec '%s = (%s / np.sum(np.abs(%s), axis=1).reshape(-1, 1))' % (data_name, data_name, data_name)
        print '%s shape: %s' % (data_name, eval('%s.shape' % data_name))

    np.savez('data/person-reid/small_triplet_viper.npz', **dict([(n, eval(n)) for n in data_list]))


def main():
    # create_mnist_triple_data('data_set/MNIST/mnist.npz')
    # create_CASIA_data('data/CASIA/CASIA-Webface_align_3/image')
    # create_single_CASIA_data('data/CASIA/src_casia.npz')

    # create_test_mnist_data('data/mnist/mnist.npz')
    # create_test_lfw_data('data/lfw/contrastive_lfw.npz')
    # create_person_reid_train_test_data('data/person-reid/viper_mix.mat')
    # create_person_reid_triplet_train_test_data('data/person-reid/viper.npz')
    create_person_reid_small_triplet_data()

    pass


if __name__ == '__main__':
    main()
