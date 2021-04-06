#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
# from tqdm import tqdm  # with .py script
from tqdm.notebook import tqdm  # with notebook


# TODO: create object to other file and inheritance
class PLSA(object):
    # TODO: add arg parser
    def __init__(self, users, items, n_class, max_iterations=10):
        self.max_iterations = max_iterations
        self.finish_ratio = 1.0e-5
        self.llh = None
        self.prev_llh = 100000.0
        self.seed = 2020

        self.users = users
        self.items = items
        self.n_users = len(set(users))
        self.n_items = len(set(items))
        self.n_data = len(users)
        self.K = n_class
        self.n_parameters = (self.K - 1) \
            + (self.K * self.n_users) \
            + (self.K * self.n_items)

        # TODO: remove test
        print('n_data: {0} | n_users: {1} | n_items: {2} | \
            n_classes: {3}'.format(
            self.n_data,
            self.n_users,
            self.n_items,
            self.K,
        ))

        np.random.seed(seed=self.seed)
        self.Pz = np.random.rand(self.K)
        self.Pu_z = np.random.rand(self.n_users, self.K)
        self.Pi_z = np.random.rand(self.n_items, self.K)

        # regularization
        self.Pz /= np.sum(self.Pz)
        self.Pu_z /= np.sum(self.Pu_z, axis=0)
        self.Pi_z /= np.sum(self.Pi_z, axis=0)

        # P(u,i,z)
        self.Puiz = np.empty((self.n_data, self.K))
        # P(z|u,i)
        self.Pz_ui = np.empty((self.n_data, self.K))

    def train(self):
        '''
        repeat e-step and m-step and predict each params
        '''
        for i in tqdm(range(self.max_iterations), desc='training plsa: '):
            self.em_algorithm()
            llh = self._calc_llh()
            ratio = abs((llh - self.prev_llh) / self.prev_llh)
            tqdm.write(" - EM: {0} | llh : {1} | ratio : {2}".format(
                i + 1,
                llh,
                ratio,
            ))

            if ratio < self.finish_ratio:
                break
            self.prev_llh = llh

        self.llh = llh

    def em_algorithm(self):
        '''
        EM algorythm
        update P(z), P(u|z), P(i|z)
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

        self.Pu_z = np.zeros((self.n_users, self.K))
        for n in range(self.n_data):
            self.Pu_z[self.users[n]] += self.Pz_ui[n]
        # (self.n_users, self.K)
        self.Pu_z = self.Pu_z / (self.n_data * self.Pz)

        self.Pi_z = np.zeros((self.n_items, self.K))
        for n in range(self.n_data):
            self.Pi_z[self.items[n]] += self.Pz_ui[n]
        # (self.n_items, self.K)
        self.Pi_z = self.Pi_z / (self.n_data * self.Pz)

    def _calc_llh(self):
        '''
        caluculate loglikelihood
        '''
        # P(u,i,z)
        self.Puiz = np.array([
            np.log(self.Pz)
            + np.log(self.Pu_z[self.users[i]])
            + np.log(self.Pi_z[self.items[i]])
            for i in range(self.n_data)
        ])  # (self.n_data, self.K)

        llh = np.sum(
            np.log(np.sum(pow(math.e, self.Puiz), axis=1)),  # (self.n_data, 1)
            axis=0,
        )
        # # TODO: remove test
        # print('P(z): {0} | llh: {1}'.format(
        #     self.Pz,
        #     llh,
        # ))
        return llh

    def get_pz_u(self):
        PzPu_z = self.Pu_z * self.Pz
        Pz_u = PzPu_z / np.sum(PzPu_z, axis=1).reshape(self.n_users, 1)
        return Pz_u

    def get_pz_i(self):
        PzPi_z = self.Pi_z * self.Pz
        Pz_i = PzPi_z / np.sum(PzPi_z, axis=1).reshape(self.n_items, 1)
        return Pz_i

    def get_aic(self):
        aic = ((-2) * self.llh) + (2 * self.n_parameters)
        return aic

    def get_bic(self):
        bic = ((-2) * self.llh) + \
            (self.n_parameters * np.log(self.n_data))
        return bic

    # TODO: make function to merge master(like index2name) and show top N items
