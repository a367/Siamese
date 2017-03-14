# -*- coding: utf-8 -*-
# @Time    : 2017/3/10 下午2:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np


class Norm2Layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        x = bottom[0].data
        # print x.shape
        # print (x[0][0]!=0)*1
        # print (x[1][0]!=0)*1
        # print (x[2][0] != 0) * 1
        # print (x[3][0] != 0) * 1
        # print 123
        # print x
        # m = x[:, 0] / x[:, 1]
        # print m
        # np.set_printoptions(threshold='nan')
        # with open('test.txt', 'w') as f:
        #     f.write(str(m))

        x_square_sum = np.sum(np.square(x), axis=1).reshape(-1, 1)
        top[0].data[...] = x / np.sqrt(x_square_sum)

        # print '---------------\n\n\n'
        #
        # print top[0].data
        # print '---------------\n\n\n'
        # print np.sqrt(x_square_sum)
        # print top[0].data[...]

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        y = top[0].data

        x_square_sum = np.sum(np.square(x), axis=1).reshape(-1, 1)
        grad = top[0].diff - y * np.sum(top[0].diff * y, axis=1).reshape(-1, 1)

        # print np.sqrt(x_square_sum)

        bottom[0].diff[...] = grad / np.sqrt(x_square_sum)

        # print bottom[0].diff[...]

        # print bottom[0].diff


class TripletLayer(caffe.Layer):
    def setup(self, bottom, top):
        # assert np.shape(bottom[0].data) == np.shape(bottom[1].data)
        # assert np.shape(bottom[0].data) == np.shape(bottom[2].data)

        # layer_params = yaml.load(self.param_str_)
        # self.margin = layer_params['margin']

        # self.margin = 0.0000001
        self.margin = 1
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        # print x_p[1]
        # print x_m[1]

        dis_p = np.sum(np.square((x - x_p)), axis=1)
        dis_m = np.sum(np.square((x - x_m)), axis=1)

        # print dis_m

        loss = dis_p - dis_m + self.margin
        self.check = (loss >= 0).reshape(-1, 1)
        # print self.check

        top[0].data[...] = np.mean(np.maximum(0, loss)) / self.margin
        # print 'loss:', np.sum(np.maximum(0, loss))

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        # print bottom[0].diff.shape
        # print np.sum(2.0 * (x_m - x_p) * self.check, axis=0).shape

        # .reshape(1, -1).repeat(bottom[0].num, axis=0)
        bottom[0].diff[...] = 2.0 * (x_m - x_p) * self.check / bottom[0].num
        bottom[1].diff[...] = 2.0 * (x_p - x) * self.check / bottom[1].num
        bottom[2].diff[...] = 2.0 * (x - x_m) * self.check / bottom[1].num


        # print bottom[1].diff[1]
        # print bottom[2].diff[2]


class TripletLayer2(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = 2
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        dis_p = np.sum(np.square((x - x_p)), axis=1)
        self.dis_m = np.sqrt(np.sum(np.square((x - x_m)), axis=1))
        loss_m = np.square(np.maximum(0, self.margin - self.dis_m))

        self.check = ((self.margin - self.dis_m) >= 0).reshape(-1, 1)

        top[0].data[...] = np.mean(loss_m + dis_p) / 2.0

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        bottom[1].diff[...] = (x_p - x) / bottom[1].num
        bottom[2].diff[...] = ((self.dis_m - self.margin) / (1e-4 + self.dis_m)).reshape(-1, 1) * (x_m - x) * self.check / bottom[2].num
        bottom[0].diff[...] = -bottom[1].diff - bottom[2].diff

        # print bottom[1].diff[1]
        # print bottom[2].diff[2]


class ContrastiveLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = 1
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_2 = bottom[1].data
        sim = bottom[2].data

        # legacy version
        # dis_p = np.sum(np.square((x - x_2)), axis=1)
        # loss_m = np.maximum(0, self.margin - dis_p)
        # self.check = (self.margin - dis_p >= 0).reshape(-1, 1)

        # current version
        dis_p = np.sum(np.square((x - x_2)), axis=1)
        self.dis_m = np.sqrt(dis_p)
        loss_m = np.square(np.maximum(0, self.margin - self.dis_m))
        self.check = (self.margin - self.dis_m >= 0).reshape(-1, 1)

        # loss
        top[0].data[...] = np.mean((1 - sim) * loss_m + sim * dis_p) / 2.0

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_2 = bottom[1].data
        sim = bottom[2].data.reshape(-1, 1)

        alpha = top[0].diff / float(bottom[0].num)

        # legacy version
        # grad_sim = (x - x_2) * alpha
        # grad_not_sim = (x_2 - x) * self.check * alpha


        # current version
        grad_sim = (x - x_2) * alpha
        grad_not_sim = ((self.dis_m - self.margin) / (self.dis_m + 1e-4)).reshape(-1, 1) * (x - x_2) * self.check * alpha

        grad = grad_sim * sim + grad_not_sim * (1 - sim)
        bottom[0].diff[...] = grad
        bottom[1].diff[...] = -grad


class OwnContrastiveLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros(bottom[0].num, dtype=np.float32)
        self.dist_sq = np.zeros(bottom[0].num, dtype=np.float32)
        self.zeros = np.zeros(bottom[0].num)
        self.m = 1.0
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        GW1 = bottom[0].data
        GW2 = bottom[1].data
        Y = bottom[2].data
        loss = 0.0
        self.diff = GW1 - GW2
        self.dist_sq = np.sum(self.diff ** 2, axis=1)
        losses = Y * self.dist_sq \
                 + (1 - Y) * np.max([self.zeros, self.m - self.dist_sq], axis=0)
        loss = np.sum(losses)
        top[0].data[0] = loss / 2.0 / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        Y = bottom[2].data
        disClose = np.where(self.m - self.dist_sq > 0.0, 1.0, 0.0)
        for i, sign in enumerate([+1, -1]):
            if propagate_down[i]:
                alphas = sign * top[0].diff[0] / bottom[i].num
                facts = (-(1 - Y) * disClose + Y) * alphas
                bottom[i].diff[...] = np.array([facts, facts]).T * self.diff
