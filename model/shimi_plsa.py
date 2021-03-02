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
        self.Puiz = np.array([
            np.log(self.Pz)
            + np.log(self.Pu_z[self.users[i]])
            + np.log(self.Pi_z[self.items[i]])
            for i in range(self.n_data)
        ])  # (self.n_data, self.K)

        # P(z|u,i)
        sum_ = np.sum(pow(math.e, self.Puiz), axis=1)  # (self.n_data,)
        self.Pz_ui = pow(math.e, self.Puiz) / sum_.reshape(self.n_data, 1)
        self.Pz_ui[np.isnan(self.Pz_ui)] = 0.0
        self.Pz_ui[np.isinf(self.Pz_ui)] = 0.0

        # M-step  -----------------------------------
        self.Pz = np.sum(self.Pz_ui, axis=0) / self.n_data

        self.Pu_z = np.zeros((self.n_uniq_users, self.K))
        for n in range(self.n_data):
            self.Pu_z[self.users[n]] += self.Pz_ui[n]
        # (self.n_uniq_users, self.K)
        self.Pu_z = self.Pu_z / (self.n_data * self.Pz)

        self.Pi_z = np.zeros((self.n_uniq_items, self.K))
        for n in range(self.n_data):
            self.Pi_z[self.items[n]] += self.Pz_ui[n]
        # (self.n_uniq_items, self.K)
        self.Pi_z = self.Pi_z / (self.n_data * self.Pz)

    def llh(self):
        '''
        対数尤度
        '''
        # P(u,i,z)
        self.Puiz = np.array([
            np.log(self.Pz)
            + np.log(self.Pu_z[self.users[i]])
            + np.log(self.Pi_z[self.items[i]])
            for i in range(self.n_data)
        ])  # (self.n_data, self.K)

        L = np.sum(
            np.log(np.sum(pow(math.e, self.Puiz), axis=1)),  # (self.n_data, 1)
            axis=0,
        )
        print('P(z): {0} | llh: {1}'.format(
            self.Pz,
            L,
        ))
        return L

    def get_Pz_u(self):
        PzPu_z = self.Pi_z * self.Pz
        Pz_u = PzPu_z / np.sum(PzPu_z, axis=1).reshape(self.n_uniq_users, 1)
        return Pz_u

    def get_Pz_i(self):
        PzPi_z = self.Pi_z * self.Pz
        Pz_i = PzPi_z / np.sum(PzPi_z, axis=1).reshape(self.n_uniq_items, 1)
        return Pz_i
