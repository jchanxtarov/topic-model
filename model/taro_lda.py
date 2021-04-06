#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from scipy.special import digamma, gammaln
# from tqdm import tqdm  # with .py script
from tqdm.notebook import tqdm  # with notebook


class LDA:
    def __init__(self, users, items, n_topics=3, max_iterations=10):
        # TODO: add arg parser
        self.max_iterations = max_iterations
        self.finish_ratio = 1.0e-5
        self.llh = None
        self.prev_llh = 100000.0
        self.perplexity = None
        # self.prev_perplexity = 100000.0  # TODO: consider
        self.seed = 2020
        np.random.seed(seed=self.seed)

        # hyper parameters
        self.alpha = 0.1
        self.beta = 0.1

        self.users = users
        self.items = items
        self.n_users = len(set(users))
        self.n_items = len(set(items))
        self.n_data = len(users)
        self.n_topics = n_topics
        self.n_parameters = (self.n_topics - 1) \
            + (self.n_topics * self.n_users) \
            + (self.n_topics * self.n_items) \
            + 2  # alpha, beta

        # TODO: remove test
        print('n_data: {0} | n_users: {1} | n_items: {2} | \
            n_topics: {3}'.format(
            self.n_data,
            self.n_users,
            self.n_items,
            self.n_topics,
        ))

        self._generate_handle_data()
        self._generate_counter_matrixes()

    def _generate_handle_data(self):
        self.dict_user_items = defaultdict(list)
        for n in range(self.n_data):
            self.dict_user_items[self.users[n]].append(self.items[n])

    def _generate_counter_matrixes(self):
        # number of times user u and topic z co-occur
        self.matrix_n_uz = np.zeros((self.n_users, self.n_topics))
        # number of times topic z and item i co-occur
        self.matrix_n_zi = np.zeros((self.n_topics, self.n_items))
        # number of times user u occur
        self.matrix_n_u = np.zeros(self.n_users)
        # number of times topic z occur
        self.matrix_n_z = np.zeros(self.n_topics)
        # topic of each user and item pair
        self.topic_ui = list(map(lambda x: np.zeros(len(x)),
                                 [self.dict_user_items[i]
                                  for i in range(self.n_users)]))

        for userid in range(len(self.dict_user_items)):
            for j in range(len(self.dict_user_items[userid])):
                itemid = self.dict_user_items[userid][j]
                topic = np.random.randint(self.n_topics)

                self.matrix_n_uz[userid, topic] += 1
                self.matrix_n_zi[topic, itemid] += 1
                self.matrix_n_u[userid] += 1
                self.matrix_n_z[topic] += 1
                self.topic_ui[userid][j] = topic

    def train(self):
        for iteration in tqdm(range(self.max_iterations)):
            self._gibbs_sampling()
            self._update_alpha()
            self._update_beta()
            self._calc_loglikelihood()

            ratio = abs((self.llh - self.prev_llh) / self.prev_llh)
            tqdm.write(" - iteration : {0} | llh : {1} | ratio : {2}".format(
                iteration + 1,
                self.llh,
                ratio,
            ))

            if ratio < self.finish_ratio:
                break
            self.prev_llh = self.llh

        self._calc_perplexity()
        print("finish training! | perplexity : {}".format(
            self.perplexity,
        ))

    def _gibbs_sampling(self):
        for i in range(len(self.dict_user_items)):
            for j in range(len(self.dict_user_items[i])):
                itemid = self.dict_user_items[i][j]

                # decrease each counters
                topic = int(self.topic_ui[i][j])
                self.matrix_n_uz[i, topic] -= 1
                self.matrix_n_zi[topic, itemid] -= 1
                self.matrix_n_z[topic] -= 1

                # increase each counters
                topic = self._sampling_topic(i, itemid)
                self.topic_ui[i][j] = topic
                self.matrix_n_uz[i, topic] += 1
                self.matrix_n_zi[topic, itemid] += 1
                self.matrix_n_z[topic] += 1

    def _sampling_topic(self, userid, itemid):
        prob = (self.matrix_n_uz[userid, :] + self.alpha) * \
            (self.matrix_n_zi[:, itemid] + self.beta) / \
            (self.matrix_n_z + self.beta * self.n_items)
        prob /= prob.sum()

        return np.random.multinomial(1, prob).argmax()

    def _update_alpha(self):
        """
        update hyperparameter alpha
        """
        numer = digamma(self.matrix_n_uz + self.alpha).sum() - \
            self.n_users * self.n_topics * digamma(self.alpha)
        denom = self.n_topics * digamma(
            self.matrix_n_u + self.alpha * self.n_topics).sum(
        ) - self.n_users * self.n_topics * digamma(
            self.alpha * self.n_topics)

        self.alpha = (self.alpha * numer) / denom

    def _update_beta(self):
        """
        update hyperparameter beta
        """
        numer = digamma(self.matrix_n_zi + self.beta).sum() - \
            self.n_topics * self.n_items * digamma(self.beta)
        denom = self.n_items * digamma(
            self.matrix_n_z + self.beta * self.n_items).sum(
        ) - self.n_topics * self.n_items * digamma(
            self.beta * self.n_items)

        self.beta = (self.beta * numer) / denom

    def _calc_loglikelihood(self):
        """
        calculate loglikelihood
        """
        # ref: https://gist.github.com/mblondel/542786#file-lda_gibbs-py-L91
        def log_multi_beta(alpha, K=None):
            if K is None:
                # alpha is assumed to be a vector
                return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
            else:
                # alpha is assumed to be a scalar
                return K * gammaln(alpha) - gammaln(K * alpha)

        llh = 0
        for z in range(self.n_topics):
            llh += log_multi_beta(self.matrix_n_zi[z, :] + self.beta)
            llh -= log_multi_beta(self.beta, self.n_items)

        for m in range(self.n_users):
            llh += log_multi_beta(self.matrix_n_uz[m, :] + self.alpha)
            llh -= log_multi_beta(self.alpha, self.n_topics)

        self.llh = llh

    # ref:
    # https://github.com/satopirka/topic-model/blob/master/lda_with_gibbs_sampling.py#L86
    def _calc_perplexity(self):
        """
        calculate perplexity
        """
        item_dist4topics = (self.matrix_n_zi + self.beta) / \
            (self.matrix_n_z[:, None] + self.beta * self.n_items)
        sum_prob = 0.0

        for i in range(len(self.dict_user_items)):
            topic_dist4users = (self.matrix_n_uz[i] + self.alpha) / (
                self.matrix_n_uz[i].sum() + self.alpha * self.n_topics)
            for j in range(len(self.dict_user_items[i])):
                topic_dist4items = item_dist4topics[:,
                                                    self.dict_user_items[i][j]]
                sum_prob += np.log(np.dot(topic_dist4items,
                                          topic_dist4users))

        self.perplexity = np.exp(- (1 / self.n_data) * sum_prob)

    def get_aic(self):
        aic = ((-2) * self.llh) + (2 * self.n_parameters)
        return aic

    def get_bic(self):
        bic = ((-2) * self.llh) + \
            (self.n_parameters * np.log(self.n_data))
        return bic

    # TODO: add functions to calculate each probability for result analysis
    # def get_pz_u(self):
    #     PzPu_z = self.Pu_z * self.Pz
    #     Pz_u = PzPu_z / np.sum(PzPu_z, axis=1).reshape(self.n_users, 1)
    #     return Pz_u

    # def get_pz_i(self):
    #     PzPi_z = self.Pi_z * self.Pz
    #     Pz_i = PzPi_z / np.sum(PzPi_z, axis=1).reshape(self.n_items, 1)
    #     return Pz_i

    # TODO: make function to merge master(like index2name) and show top N items
