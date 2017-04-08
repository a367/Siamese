# -*- coding: utf-8 -*-
# @Time    : 2017/4/7 下午3:06
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np
import random
import config


class DataLayer(caffe.Layer):
    def get_next_minibatch(self):
        train_X = []
        train_Y = []
        sample_big_ids = random.sample(self.big_ids, 5)

        for pid in sample_big_ids:
            face_num = len(self.train_imgs[pid])
            select_num = random.randint(config.MIN_IDENTITY_SIZE, min(config.MAX_IDENTITY_SIZE, face_num))
            sample_face_ids = random.sample(range(face_num), select_num)
            train_X += [self.train_imgs[pid][fid] for fid in sample_face_ids]
            train_Y += [pid] * select_num

        rest_num = config.BATCH_SIZE - len(train_X)
        for i in xrange(rest_num):
            # random produce pid, which not in big ids
            pid = random.randint(0, self.person_num - 1)
            while pid in sample_big_ids:
                pid = random.randint(0, self.person_num - 1)

            train_X.append(random.choice(self.train_imgs[pid]))
            train_Y.append(pid)

        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        select_X = (train_X - config.MEAN_VALUE) * config.SCALE
        select_Y = train_Y

        return {'data': select_X, 'label': select_Y}

    def setup(self, bottom, top):
        src = np.load(config.DATA_PATH)

        # set train_imgs to memory
        self.train_imgs = src['X']
        self.person_num = self.train_imgs.shape[0]
        self.big_ids = [i if len(self.train_imgs[i]) >= config.MIN_IDENTITY_SIZE else None for i in xrange(len(self.train_imgs))]
        self.big_ids = filter(lambda x: x != None, self.big_ids)
        self.big_num = len(self.big_ids)

        # (batch_size, channels, weight, height)
        top[0].reshape(config.BATCH_SIZE, 1, 96, 112)
        top[1].reshape(config.BATCH_SIZE)

    def forward(self, bottom, top):
        blobs = self.get_next_minibatch()

        top[0].data[...] = blobs['data']
        top[1].data[...] = blobs['label']

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
