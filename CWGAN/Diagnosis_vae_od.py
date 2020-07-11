# -*- coding: utf-8 -*-
"""
Schindler Liang
MLP Variational AutoEncoder for Anomaly Detection
reference: https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf
Created on  June  5 10:54:44 2019
@author: Jianye Su
"""
import random
import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from option import Options


op = Options()
level_num = op.level_num
select_number = op.select_number


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x)


def build_dense(input_vector, unit_no, activation):
    return tf.layers.dense(input_vector, unit_no, activation=activation,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer())


class MLP_VAE:
    def __init__(self, input_dim, lat_dim, outliers_fraction):
        # input_paras:
        # input_dim: input dimension for X
        # lat_dim: latent dimension for Z
        # outliers_fraction: pre-estimated fraction of outliers in trainning dataset

        self.outliers_fraction = outliers_fraction  # for computing the threshold of anomaly score
        self.input_dim = input_dim
        self.lat_dim = lat_dim  # the lat_dim can exceed input_dim？

        self.input_X = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='source_x')

        self.learning_rate = 0.0005
        self.batch_size = 32
        # batch_size should be smaller than normal setting for getting
        # a relatively lower anomaly-score-threshold
        self.train_iter = 10000 #10000
        self.hidden_units = 128
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0

    def _encoder(self,):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X, self.hidden_units, activation=lrelu)
            l2 = build_dense(l1, self.hidden_units, activation=lrelu)
            mu = tf.layers.dense(l2, self.lat_dim)
            sigma = tf.layers.dense(l2, self.lat_dim, activation=tf.nn.softplus)
            sole_z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        return mu, sigma, sole_z

    def _decoder(self, z):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            l1 = build_dense(z, self.hidden_units, activation=lrelu)
            l2 = build_dense(l1, self.hidden_units, activation=lrelu)
            recons_X = tf.layers.dense(l2, self.input_dim)
        return recons_X

    def _build_VAE(self):
        self.mu_z, self.sigma_z, sole_z = self._encoder()
        self.recons_X = self._decoder(sole_z)

        with tf.variable_scope('loss'):
            KL_divergence = 0.5 * tf.reduce_sum(
                tf.square(self.mu_z) + tf.square(self.sigma_z) - tf.log(1e-8 + tf.square(self.sigma_z)) - 1, 1)
            mse_loss = tf.reduce_sum(tf.square(self.input_X - self.recons_X), 1)
            self.all_loss = mse_loss
            self.loss = tf.reduce_mean(mse_loss + KL_divergence)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _fecth_data(self, input_data):
        if (self.pointer + 1) * self.batch_size >= input_data.shape[0]:
            return_data = input_data[self.pointer * self.batch_size:, :]
            self.pointer = 0
        else:
            return_data = input_data[self.pointer * self.batch_size:(self.pointer + 1) * self.batch_size, :]
            self.pointer = self.pointer + 1
        return return_data

    def train(self, train_X):
        for index in range(self.train_iter):
            this_X = self._fecth_data(train_X)
            self.sess.run([self.train_op], feed_dict={
                self.input_X: this_X
            })
        self.arrage_recons_loss(train_X)

    def arrage_recons_loss(self, input_data):
        all_losses = self.sess.run(self.all_loss, feed_dict={
            self.input_X: input_data
        })
        self.judge_loss = np.percentile(all_losses, (1 - self.outliers_fraction) * 100)


    def judge(self, input_data):
        return_label = []
        anomaly_num = 0
        for index in range(input_data.shape[0]):
            single_X = input_data[index].reshape(1, -1)
            this_loss = self.sess.run(self.loss, feed_dict={
                self.input_X: single_X
            })
            if this_loss < self.judge_loss:
                return_label.append(1)
            else:
                return_label.append(-1)
                anomaly_num += 1
        return anomaly_num


def mlp_vae_predict(train, test):
    mlp_vae = MLP_VAE(65, 20, 0.07)
    mlp_vae.train(train)
    anomaly_num = mlp_vae.judge(test)
    return anomaly_num


def get_min_max():
    train_data = np.loadtxt("train_data"+str(select_number)+"_level"+str(level_num)+"_original.txt")
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)
    return min, max


def MaxMinNormalization(x, min, max):
    x = (x - min) / (max - min + 0.0000001)
    return x


def load_data(select_num, min, max):

    size = select_num * 7
    total_num = 36337
    step_size = 5191
    TR_sample_temp = sio.loadmat("Level"+str(level_num)+".mat")
    sample = TR_sample_temp['num']

    # Select sample
    for i in range(0, total_num, step_size):
        num = random.sample(range(i, step_size + i), select_num)
        if i == 0:
            train_data = sample[num[0]]
        temp = i
        for j in num:
            if temp == 0:
                temp = -1
                continue
            else:
                train_data = np.row_stack((train_data, sample[j]))
    train_labels = train_data[:, 0].reshape(size, 1)
    train_data = np.delete(train_data, [0], axis=1)
    train_data = MaxMinNormalization(train_data, min, max)
    return train_data, train_labels


def choose():
    # Load generated data
    train_data = np.loadtxt('CWGAN_data' + str(select_number) + '_level' + str(level_num) +'.txt')
    test_data = np.loadtxt("train_data"+str(select_number)+"_level"+str(level_num)+"_normalization.txt")
    train_data_ = np.split(train_data, 7)
    test_data_ = np.vsplit(test_data, 7)
    filter_n = 10
    for i in range(7):
        result_temp_all = []
        for j in range(filter_n):
            print("Class:%d/7  Filter_n:%d/%d" % (i+1, j+1, filter_n))
            num = random.sample(range(0, 1500), 1000)
            train_data_temp = train_data_[i][num]
            test_data_temp = test_data_[i]
            result_temp = mlp_vae_predict(train_data_temp, test_data_temp)
            if j == 0:
                train_data_temp_all = train_data_temp
            else:
                train_data_temp_all = np.row_stack((train_data_temp_all, train_data_temp))
            result_temp_all.append(result_temp)
            print("temp anomaly numbers:", result_temp_all, '\n')
        min_index = result_temp_all.index(min(result_temp_all))
        train_data_temp_all = np.reshape(train_data_temp_all, (filter_n, 1000, 65))
        if i == 0:
            train_data_select = train_data_temp_all[min_index]
        else:
            train_data_select = np.row_stack((train_data_select, train_data_temp_all[min_index]))
        print("Current filtering dimension：", train_data_select.shape, "Target dimension：（7000， 65）")
    np.savetxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_vae_od.txt", train_data_select)
    return train_data_select


def classify(x_train1, y_train1, iteration):
    min, max = get_min_max()
    x_test, y_test = load_data(500, min, max)

    # Random Forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(x_train1, y_train1)
    RF_pre = rfc1.predict(x_test)
    RF_AC = accuracy_score(y_test, RF_pre)
    RF_f1 = f1_score(y_test, RF_pre, average='macro')

    # SVM
    # print("Phase 2 SVM parameters selecting...")
    # parameters = {
    #     'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #     'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'kernel': ['rbf'],
    #     'decision_function_shape': ['ovr']
    # }
    # svc = svm.SVC()
    # grid_search = GridSearchCV(svc, parameters, scoring='accuracy', cv=5)
    # grid_search.fit(train_data, train_label.ravel())
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print("hello:", para, val)
    # clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True).fit(
    #     train_data, train_label.ravel())

    # SVM
    clf = SVC(kernel='rbf', C=9, gamma=0.1)
    clf.set_params(kernel='rbf', probability=True).fit(x_train1, y_train1)
    clf.predict(x_train1)
    test_pre = clf.predict(x_test)
    SVM_AC = accuracy_score(y_test, test_pre)
    SVM_f1 = f1_score(y_test, test_pre, average='macro')

    # decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train1, y_train1)
    dt_pre = dtc.predict(x_test)
    DT_AC = accuracy_score(y_test, dt_pre)
    DT_f1 = f1_score(y_test, dt_pre, average='macro')

    # Bayes
    mnb = MultinomialNB()
    mnb.fit(x_train1, y_train1)
    NB_predict = mnb.predict(x_test)
    NB_AC = accuracy_score(y_test, NB_predict)
    NB_f1 = f1_score(y_test, NB_predict, average='macro')

    # Multilayer perceptron
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(x_train1, y_train1)
    MLP_predict = MLP.predict(x_test)
    MLP_AC = accuracy_score(y_test, MLP_predict)
    MLP_f1 = f1_score(y_test, MLP_predict, average='macro')

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train1, y_train1)
    knn_predict = knn.predict(x_test)
    KNN_AC = accuracy_score(y_test, knn_predict)
    KNN_f1 = f1_score(y_test, knn_predict, average='macro')

    # LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(x_train1, y_train1)
    lg_predict = classifier.predict(x_test)
    LG_AC = accuracy_score(y_test, lg_predict)
    LG_f1 = f1_score(y_test, lg_predict, average='macro')
    print("===== Diagnosis vae_od evaluation %d / 1=======" % (iteration+1))
    print('vae_od Accuracy:')
    print(RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC)
    print('F1-score')
    print(RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1)
    file_name1 = "./temp_result/Diagnosis_" + str(select_number) + "Level" + str(level_num) + "_Accuracy_result.txt"
    file_name2 = "./temp_result/Diagnosis_" + str(select_number) + "Level" + str(level_num) + "_f1_score_result.txt"
    with open(file_name1, "a") as f:
        f.writelines(
            [str(RF_AC), ' ', str(SVM_AC), ' ', str(DT_AC), ' ', str(NB_AC), ' ', str(MLP_AC), ' ', str(KNN_AC), ' ',
             str(LG_AC), '\n'])
    with open(file_name2, "a") as f:
        f.writelines(
            [str(RF_f1), ' ', str(SVM_f1), ' ', str(DT_f1), ' ', str(NB_f1), ' ', str(MLP_f1), ' ', str(KNN_f1), ' ',
             str(LG_f1), '\n'])
    return RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC, RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


def evaluation():
    x_train1 = choose()
    y_train1 = np.ones((1000, 1))
    init = 1
    for i in range(6):
        temp = np.ones((1000, 1)) + init
        y_train1 = np.row_stack((y_train1, temp))
        init += 1
    y_train1 = y_train1.ravel()
    for i in range(1):
        RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC, RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1\
            = classify(x_train1, y_train1, i)
        if i == 0:
            ave_RF = RF_AC
            ave_SVM = SVM_AC
            ave_DT = DT_AC
            ave_NB = NB_AC
            ave_MLP = MLP_AC
            ave_KNN = KNN_AC
            ave_LG = LG_AC

            ave_RF_f1 = RF_f1
            ave_SVM_f1 = SVM_f1
            ave_DT_f1 = DT_f1
            ave_NB_f1 = NB_f1
            ave_MLP_f1 = MLP_f1
            ave_KNN_f1 = KNN_f1
            ave_LG_f1 = LG_f1
        else:
            ave_RF = np.append(ave_RF, RF_AC)
            ave_SVM = np.append(ave_SVM, SVM_AC)
            ave_DT = np.append(ave_DT, DT_AC)
            ave_NB = np.append(ave_NB, NB_AC)
            ave_MLP = np.append(ave_MLP, MLP_AC)
            ave_KNN = np.append(ave_KNN, KNN_AC)
            ave_LG = np.append(ave_LG, LG_AC)

            ave_RF_f1 = np.append(ave_RF_f1, RF_f1)
            ave_SVM_f1 = np.append(ave_SVM_f1, SVM_f1)
            ave_DT_f1 = np.append(ave_DT_f1, DT_f1)
            ave_NB_f1 = np.append(ave_NB_f1, NB_f1)
            ave_MLP_f1 = np.append(ave_MLP_f1, MLP_f1)
            ave_KNN_f1 = np.append(ave_KNN_f1, KNN_f1)
            ave_LG_f1 = np.append(ave_LG_f1, LG_f1)


evaluation()

