#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import math
import sys
import time

import numpy as np
import pandas as pd
# import slackweb
import requests
from sklearn.preprocessing import StandardScaler
from tqdm import trange

# ************
# parameter
# ************
URL_slack = 'https://hooks.slack.com/services/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'
PATH_data = '/Users/ryotaro/Desktop/現在の研究/OdaQ_APIEM/savedata/'
PATH_save = '/Users/ryotaro/Desktop/現在の研究/OdaQ_APIEM/main/method1/purchasing_behaviour/'

data_time_now = '2018-07-24-01-24'
result_time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

max_repeat_num = 10
finish_ratio = 1.0e-3
slack_announce = False
# list_for_Z = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
list_for_Z = [8]

DBL_MIN = sys.float_info.min
DBL_MAX = sys.float_info.max


class PLSA(object):
    def __init__(self, bow_count_shop, K):
        # self.card = card
        self.bow_count_shop = bow_count_shop
        # self.count = np.log10(count)
        self.K = Z
        self.C = len(bow_count_shop)
        self.S = len(bow_count_shop[0])
        # self.n_data = len(card)
        # print('data number is', self.n_data)
        print('len C : ', self.C)
        print('len S : ', self.S)

        # P(z)
        self.Pz = np.random.rand(self.K)
        self.Pz /= np.sum(self.Pz)

        # P(s|z)
        self.Ps_z = np.random.rand(self.S, self.K)
        self.Ps_z /= np.sum(self.Ps_z, axis=0)

        # self.mu_s = np.random.rand(self.S, self.K) * 1
        # self.sigma_s = np.random.rand(self.S, self.K) * 10

        # P(data, z)
        self.Pall = np.zeros((self.C, self.K))
        # P(z|data)
        self.Pz_data = np.zeros((self.C, self.K))
        # P(data)
        self.Pdata = np.zeros(self.C)

    def train(self, max_repeat_num, finish_ratio):
        '''
        対数尤度が収束するまでEステップとMステップを繰り返す
        '''
        prev_llh = 100000.0
        em_start = time.time()
        Pall = 0  # memo : kari

        for i in range(max_repeat_num):
            print('em', i + 1, '回')
            self.em_algorithm(i, Pall)
            llh, Pall = self.llh()
            print("llh : ", llh)

            em_time_now = time.time() - em_start
            hour, minute, second = em_time_now / \
                3600, (em_time_now % 3600)/60, em_time_now % 60
            print("K = ", self.K, '\nem', i + 1, '回', "\nlap : ",
                  int(hour), "-", int(minute), "-", int(second))
            print("ratio : ", abs((llh - prev_llh) / prev_llh))

            if slack_announce:
                post_slack = {
                    'text': "K = " + str(self.K) + '\n em' + str(i+1) + '回' + "\nratio : " + str(abs((llh - prev_llh) / prev_llh)) + "\nllh : " + str(llh) + "\n lap : " + str(int(hour)) + "-" + str(int(minute)) + "-" + str(int(second)),
                    'username': 'report : plsa_POS',
                    'link_names': 1
                }
                requests.post(URL_slack, data=json.dumps(post_slack))

            if abs((llh - prev_llh) / prev_llh) < finish_ratio:
                break
            prev_llh = llh
        return llh

    def gauss_distribution(self, count, mu, sigma):
        P = (1.0 / np.sqrt(2.0 * np.pi * sigma)) * \
            np.exp(-((count - mu)**2) / (2 * sigma))
        return P

    def em_algorithm(self, count, Pall):
        '''
        EMアルゴリズム
        P(z), P(c|z), P(s|z), P(g|z), P(r|z)の更新
        '''
        # E-step  -----------------------------------
        # P(all)
        if count == 0:
            # P(data, z)
            self.Pall = np.ones((self.C, self.K))
            for c in range(self.C):
                self.Pall[c] *= self.Pz
                self.Pall[c] *= np.prod(pow(self.Ps_z.T,
                                            self.bow_count_shop[c]), axis=1)
                self.Pall[c] *= np.prod(pow((1.0 - self.Ps_z).T,
                                            1 - self.bow_count_shop[c]), axis=1)
                # self.Pall[self.card[n]] *= self.Pz
                # self.Pall[self.card[n]] *= self.gauss_distribution(self.bow_count_shop[c], self.mu_s, self.sigma_s)
            # self.Pall = self.Pall / np.sum(self.Pall)
            # print(self.Pall)
            # print(np.sum(self.Pall))

        else:
            self.Pall = Pall

        self.Pall[np.isnan(self.Pall)] = 0
        self.Pall[np.isinf(self.Pall)] = 0

        # P(z|data)
        self.Pz_data = np.zeros((self.C, self.K))
        for c in range(self.C):
            # print(self.Pall[c])
            # print(np.sum(self.Pall[c]))
            self.Pz_data[c] = self.Pall[c] / np.sum(self.Pall[c])
            # print(self.Pz_data[c])

        self.Pz_data[np.isnan(self.Pz_data)] = 0
        self.Pz_data[np.isinf(self.Pz_data)] = 0

        # M-step  -----------------------------------
        # P(z)
        for k in range(self.K):
            self.Pz[k] = np.sum(self.Pz_data[:, k]) / np.sum(self.Pz_data)
        print('P(z) : ', self.Pz)

        # ToDo : ここ確認したい
        # binary_distribution
        self.Ps_z = np.zeros((self.S, self.K))
        bunbo = np.sum(self.Pz_data, axis=0)
        self.Ps_z = np.dot(self.bow_count_shop.T, self.Pz_data) / bunbo
        # self.Ps_z /= np.sum(self.Ps_z, axis=0)
        # print('test : ', np.sum(self.Ps_z, axis = 0))

        # # ToDo : 確認必要
        # # gauss_distribution
        # self.mu_s = np.zeros((self.S, self.K))
        # self.sigma_s = np.zeros((self.S, self.K))
        # bunbo = np.sum(self.Pz_data, axis = 0)
        # for s in range(self.S):
        #     self.mu_s[s] = (self.Pz_data * self.bow_count_shop[s]) / bunbo
        #     self.sigma_s[s] += (self.Pz_data * ((self.bow_count_shop[s] - self.mu_s[s])**2)) / bunbo

    def llh(self):
        '''
        対数尤度
        '''
        # P(data, z)
        self.Pall = np.ones((self.C, self.K))
        for c in range(self.C):
            self.Pall[c] *= self.Pz
            self.Pall[c] *= np.prod(pow(self.Ps_z.T,
                                        self.bow_count_shop[c]), axis=1)
            self.Pall[c] *= np.prod(pow((1.0 - self.Ps_z).T,
                                        1 - self.bow_count_shop[c]), axis=1)
            # print(self.Pall[c])
            # self.Pall[self.card[n]] *= self.Pz
            # self.Pall[self.card[n]] *= self.gauss_distribution(self.bow_count_shop[c], self.mu_s, self.sigma_s)
        # self.Pall = self.Pall / np.sum(self.Pall)

        self.Pall[np.isnan(self.Pall)] = DBL_MIN
        self.Pall[np.isinf(self.Pall)] = DBL_MAX

        L = 0
        a = 0
        for c in range(self.C):
            # Prus = np.sum(pow(math.e, self.Pall[c]))
            Prus = np.sum(self.Pall[c])
            if np.isnan(np.log(Prus)) or np.isinf(np.log(Prus)):
                L += 0
                a += 1
            else:
                L += np.log(Prus)
        print('count_nan_main : ', a)
        return L, self.Pall


if __name__ == '__main__':
    df = pd.read_csv(PATH_data + "POS_for_plsa_poisson_" +
                     data_time_now + ".csv")
    # df = df[;, ['card', 'shop', 'count']].sort_values('card').reset_index(drop = True)
    df = df.loc[:, ['new_shop_id', 'new_card_id', 'count']
                ].sort_values('new_card_id').reset_index(drop=True)
    # df_POS_bow = df.pivot_table(values = ['count'], index = ['new_card_id'], columns = ['new_shop_id'], aggfunc = 'sum', fill_value = 0)

    array_new_card_id = np.array(df['new_card_id'].values)
    array_new_shop_id = np.array(df['new_shop_id'].values)
    array_count = np.array(df['count'].values)
    print('size :', np.sum(array_count))

    S = len(df['new_shop_id'].unique())
    C = len(df['new_card_id'].unique())

    print('start get bow !')
    array_temp = np.zeros((C, S), dtype='int')

    for i in trange(len(array_new_card_id)-1):
        array_temp[array_new_card_id[i]][array_new_shop_id[i]] = 1
        # list_temp[array_new_card_id[i]][array_new_shop_id[i]] = array_count[i]

    # card = list(df_POS_bow['new_card_id'].values)
    bow_count_shop = array_temp
    # np.savetxt(PATH_data + 'bow_onehot_count_shop_' + data_time_now + ".csv", bow_count_shop, delimiter=',')
    # np.savetxt(PATH_data + 'bow_count_shop_' + date_time_now + ".csv", bow_count_shop, delimiter=',')

    # # 標準化
    # for target in list(df.columns):
    # df['count'] = list(StandardScaler().fit_transform(df['count'].reshape(-1, 1)))

    print('complete prep!')
    list_AIC = []
    list_BIC = []
    list_Z = []
    list_LL = []

    for Z in list_for_Z:
        # try:
        K = Z
        parameter = (Z-1) + Z*(len(bow_count_shop[0]))

        time_start = time.time()
        print('************ plsa start ************')
        print('class is ', Z)
        plsa = PLSA(bow_count_shop, Z)
        LL = plsa.train(max_repeat_num, finish_ratio)

        AIC = ((-2)*LL)+(2*parameter)
        BIC = ((-2)*LL)+(parameter*np.log(len(bow_count_shop)))
        print('train finished, LL= ', LL)

        plsa_all_time = time.time() - time_start
        hour, minute, second = plsa_all_time / \
            3600, (plsa_all_time % 3600)/60, plsa_all_time % 60
        print("K = ", Z, "\ntime : ", int(hour), "-",
              int(minute), "-", int(second), " (finished!)")

        if slack_announce:
            post_slack = {
                'text': "K = " + str(Z) + "\ntime : " + str(int(hour)) + "-" + str(int(minute)) + "-" + str(int(second)) + " (finished!)",
                #     'channel' : 'gotolab',
                'username': 'report : plsa_POS',
                'link_names': 1
            }
            requests.post(URL_slack, data=json.dumps(post_slack))

        print('AIC = ', AIC, ', BIC = ', BIC)
        list_AIC.append(AIC)
        list_BIC.append(BIC)
        list_LL.append(LL)
        list_Z.append(Z)

        time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        np.savetxt(PATH_save + 'purchasing_behaviour_z' + str(Z) +
                   '_pz_' + time_now + ".csv", plsa.Pz, delimiter=',')
        np.savetxt(PATH_save + 'purchasing_behaviour_z' + str(Z) +
                   '_Ps_z_' + time_now + ".csv", plsa.Ps_z, delimiter=',')
        # np.savetxt(PATH_save + 'purchasing_behaviour_z' + str(Z) + '_mu_shop_' + time_now + ".csv", plsa.mu_s, delimiter=',')
        # np.savetxt(PATH_save + 'purchasing_behaviour_z' + str(Z) + '_sigma_shop_' + time_now + ".csv", plsa.sigma_s, delimiter=',')
        np.savetxt(PATH_save + 'purchasing_behaviour_z' + str(Z) +
                   '_Pz_data_' + time_now + ".csv", plsa.Pz_data, delimiter=',')
        pd.DataFrame({"Z": list_Z, "LL": list_LL, "AIC": list_AIC, 'BIC': list_BIC}).to_csv(
            PATH_save + "purchasing_behaviour_Z" + str(Z) + "_LL_AIC_" + time_now + ".csv", index=None)

        # except:
        #     # alert_error
        #     if slack_announce:
        #         post_slack = {
        #             'text': "errorだよん。", 'username': 'em_announcementだよん！', 'link_names': 1
        #             }
        #         requests.post(URL_slack, data = json.dumps(post_slack))
