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
