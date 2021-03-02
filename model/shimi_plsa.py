#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
from tqdm import tqdm


class PLSA(object):
    # TODO: arg parser の導入
    def __init__(self, users, items, n_class, max_repeat_num=10):
        self.max_repeat_num = max_repeat_num
        self.finish_ratio = 1.0e-5
        self.prev_llh = 100000.0

        self.users = users
        self.items = items
        self.n_uniq_users = len(set(users))
        self.n_uniq_items = len(set(items))
        self.n_data = len(users)
        self.K = n_class
        print('n_data: {0} | n_uniq_users: {1} | n_uniq_items: {2} | n_class: {3}'.format(
            self.n_data,
            self.n_uniq_users,
            self.n_uniq_items,
            self.K,
        ))

        self.Pz = np.random.rand(self.K)
        self.Pu_z = np.random.rand(self.n_uniq_users, self.K)
        self.Pi_z = np.random.rand(self.n_uniq_items, self.K)

        # 正規化
        self.Pz /= np.sum(self.Pz)
        self.Pu_z /= np.sum(self.Pu_z, axis=0)
        self.Pi_z /= np.sum(self.Pi_z, axis=0)

        # P(u,i,z)
        self.Puiz = np.empty((self.n_data, self.K))
        # P(z|u,i)
        self.Pz_ui = np.empty((self.n_data, self.K))

    def train(self):
        '''
        対数尤度が収束するまでEステップとMステップを繰り返す
        '''
        for i in tqdm(range(self.max_repeat_num)):
            print('EM: {}'.format(i + 1))
            self.em_algorithm()
            llh = self.llh()
            ratio = abs((llh - self.prev_llh) / self.prev_llh)
            print("llh : {0} | ratio : {1}".format(
                llh,
                ratio,
            ))

            if ratio < self.finish_ratio:
                break
            self.prev_llh = llh

        return llh

    def em_algorithm(self):
        '''
        EMアルゴリズム
        P(z), P(u|z), P(i|z)の更新
        '''
        # E-step  -----------------------------------
        # P(u,i,z)
        # TODO: 二重ループの回避
        for k in range(self.K):
            for i in range(self.n_data):
                self.Puiz[i][k] = np.log(self.Pz[k]) \
                    + np.log(self.Pu_z[self.users[i]][k]) \
                    + np.log(self.Pi_z[self.items[i]][k])

        # P(z|u,i)
        # TODO: 二重ループの回避
        for i in range(self.n_data):
            sum_ = 0.0
            for k in range(self.K):
                sum_ += pow(math.e, self.Puiz[i][k])
            for k in range(self.K):
                self.Pz_ui[i][k] = pow(math.e, self.Puiz[i][k]) / sum_

        self.Pz_ui[np.isnan(self.Pz_ui)] = 0.0
        self.Pz_ui[np.isinf(self.Pz_ui)] = 0.0

        # M-step  -----------------------------------
        self.Pz = np.sum(self.Pz_ui, axis=0) / self.n_data

        self.Pu_z = np.zeros((self.n_uniq_users, self.K))
        # TODO: 二重ループの回避
        for k in range(self.K):
            for i in range(self.n_data):
                self.Pu_z[self.users[i]][k] += self.Pz_ui[i][k]
            for i in range(self.n_uniq_users):
                self.Pu_z[i][k] = self.Pu_z[i][k] / (self.n_data * self.Pz[k])

        self.Pi_z = np.zeros((self.n_uniq_items, self.K))
        for k in range(self.K):
            for i in range(self.n_data):
                self.Pi_z[self.items[i]][k] += self.Pz_ui[i][k]
            for i in range(self.n_uniq_items):
                self.Pi_z[i][k] = self.Pi_z[i][k] / (self.n_data * self.Pz[k])

    def llh(self):
        '''
        対数尤度
        '''
        # P(u,i,z)
        # TODO: 二重ループの回避
        for k in range(self.K):
            for i in range(self.n_data):
                self.Puiz[i][k] = np.log(self.Pz[k]) \
                    + np.log(self.Pu_z[self.users[i]][k]) \
                    + np.log(self.Pi_z[self.items[i]][k])

        # P(z|u,i)
        L = 0
        # TODO: 二重ループの回避
        for i in range(self.n_data):
            sum_ = 0
            for k in range(self.K):
                sum_ += pow(math.e, self.Puiz[i][k])
            L += np.log(sum_)
        print('P(z): {0} | llh: {1}'.format(
            self.Pz,
            L,
        ))
        return L
