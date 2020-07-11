# 使用异常检测方法筛选高质量生成样本
# 首先将某类生成的1500个故障样本随机选取1000个作为VAE 的训练数据，训练样本x输入到编码器后，编码器输出均值和方差参数组成潜在变量z。
# 之后z输入到解码器，解码器输出重建样本x’，进而计算x’与x的重建损失MSE。使用编码器输出的均值和方差参数，计算该分布与先验分布的KL散度。
# 总的损失定义重建损失与KL散度的加和。训练完成之后输入全部样本计算平均的重建损失，
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

level_num = 1
select_number = 10
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
        self.input_dim = input_dim  # 输入维度
        self.lat_dim = lat_dim  # the lat_dim can exceed input_dim？？？？？？

        self.input_X = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='source_x')

        self.learning_rate = 0.0005
        self.batch_size = 32
        # batch_size should be smaller than normal setting for getting
        # a relatively lower anomaly-score-threshold 批量大小应小于正常设置，以获得相对较低的异常分数阈值
        self.train_iter = 10000 #10000
        self.hidden_units = 128
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0

    def _encoder(self,):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X, self.hidden_units, activation=lrelu)
            #            l1 = tf.nn.dropout(l1,0.8)
            l2 = build_dense(l1, self.hidden_units, activation=lrelu)
            #            l2 = tf.nn.dropout(l2,0.8)
            mu = tf.layers.dense(l2, self.lat_dim)  # 均值向量
            sigma = tf.layers.dense(l2, self.lat_dim, activation=tf.nn.softplus)  # 标准差向量
            sole_z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)  # 潜在变量z
        return mu, sigma, sole_z

    def _decoder(self, z):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            l1 = build_dense(z, self.hidden_units, activation=lrelu)
            #            l1 = tf.nn.dropout(l1,0.8)
            l2 = build_dense(l1, self.hidden_units, activation=lrelu)
            #            l2 = tf.nn.dropout(l2,0.8)
            recons_X = tf.layers.dense(l2, self.input_dim)  # 重建的x
        return recons_X

    def _build_VAE(self):
        self.mu_z, self.sigma_z, sole_z = self._encoder()
        # print("mu_z shape:", self.mu_z.shape)
        self.recons_X = self._decoder(sole_z)

        with tf.variable_scope('loss'):
            #  KL计算方式 KL(N（mu,sigma^2）, N(0,1)) = 0.5*（mu^2+sigma^2-log(sigma^2)-1）
            # print(self.mu_z.shape, self.sigma_z.shape)
            # 每个样本的KL散度，结果为一维数组shape（batch，）
            KL_divergence = 0.5 * tf.reduce_sum(
                tf.square(self.mu_z) + tf.square(self.sigma_z) - tf.log(1e-8 + tf.square(self.sigma_z)) - 1, 1)
            # print("KL_divergence shape:", KL_divergence.shape)
            # 每个样本的重建损失，结果为一维数组shape（batch，65）
            mse_loss = tf.reduce_sum(tf.square(self.input_X - self.recons_X), 1)
            # print("mse_loss:", mse_loss)
            self.all_loss = mse_loss
            # 计算该batch的平均损失（重建损失+KL散度）
            self.loss = tf.reduce_mean(mse_loss + KL_divergence)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _fecth_data(self, input_data):
        if (self.pointer + 1) * self.batch_size >= input_data.shape[0]:  # 下一个batchsize超出样本范围
            return_data = input_data[self.pointer * self.batch_size:, :]  # 取出剩余的样本并重置pointer
            self.pointer = 0
        else:
            return_data = input_data[self.pointer * self.batch_size:(self.pointer + 1) * self.batch_size, :]
            self.pointer = self.pointer + 1
        return return_data

    def train(self, train_X):
        for index in range(self.train_iter):
            this_X = self._fecth_data(train_X)  # train_X中的batchsize输入
            self.sess.run([self.train_op], feed_dict={  # 最小化kl散度加重建损失
                self.input_X: this_X
            })
        #  训练完之后只计算总共样本的重建损失，该重建损失用于测试
        self.arrage_recons_loss(train_X)

    def arrage_recons_loss(self, input_data):
        # 这次的all_loss是全部训练集放进去计算重建损失以便选择异常阈值，以前的是一个batchsize
        # 结果为一维数组（1000，）
        all_losses = self.sess.run(self.all_loss, feed_dict={
            self.input_X: input_data
        })
        #  numpy.percentile(a, p, axis)一个多维数组的任意百分比分位数(先升序排列)
        # 返回一个数值 如p=50代表a里排序之后的中位数  p:要计算的百分位数，在 0 ~ 100 之间
        # 第p个百分位数是这样一个值，它使得至少有p%的数据项小于或等于这个值，且至少有(100-p)%的数据项大于或等于这个值。此处p=93
        # 该语句即选择重建损失阈值，all_losses为一维数组内容，为各个样本的重建损失，升序排列后选择第93%的损失
        self.judge_loss = np.percentile(all_losses, (1 - self.outliers_fraction) * 100)


    def judge(self, input_data):
        return_label = []
        anomaly_num = 0
        for index in range(input_data.shape[0]):
            single_X = input_data[index].reshape(1, -1)
            #  计算单个测试样本总的loss即kl和重建损失之和
            this_loss = self.sess.run(self.loss, feed_dict={
                self.input_X: single_X
            })
            #  judge_loss为重建损失  总的loss小于重建loss的阈值?我觉得应该是测试样本的重建loss小于重建loss的阈值
            if this_loss < self.judge_loss:  # 正常数据
                return_label.append(1)
            else:
                return_label.append(-1)
                anomaly_num += 1
        return anomaly_num


# 返回异常样本个数
def mlp_vae_predict(train, test):
    mlp_vae = MLP_VAE(65, 20, 0.07)  # input_dim, lat_dim, outliers_fraction
    mlp_vae.train(train)
    anomaly_num = mlp_vae.judge(test)
    return anomaly_num
    #plot_confusion_matrix(test_label, mlp_vae_predict_label, ['anomaly', 'normal'], 'MLP_VAE Confusion-Matrix')


#  获取训练集的最小值和最大值，测试集的min-max归一化使用训练集的最小值与最大值
def get_min_max():
    train_data = np.loadtxt("train_data"+str(select_number)+"_level"+str(level_num)+"_original.txt")
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)
    return min, max


def MaxMinNormalization(x, min, max):
    x = (x - min) / (max - min + 0.0000001)
    return x


# 加载真实故障数据为了30次测试
def load_data(select_num, min, max):  # select_num = 50 #选择每类选取的故障样本数

    size = select_num * 7   # 共7类
    total_num = 36337
    step_size = 5191
    TR_sample_temp = sio.loadmat("Level"+str(level_num)+".mat")
    sample = TR_sample_temp['num']  # 每类5191个，故数据总量为36337*66，第一列为类标

    # Select sample
    # 从每个区间中（共7个区间，每个区间5191个样本）随机获取t个（50个）元素，作为一个片断返回（共250个）
    for i in range(0, total_num, step_size):
        num = random.sample(range(i, step_size + i), select_num)  # 返回随机选中的样本编号组成的数组
        if i == 0:
            train_data = sample[num[0]]  # 初始化train_data,方便以后的np.row_stack
        temp = i
        for j in num:
            if temp == 0:  # 跳过已加入的sample[num[0]]
                temp = -1
                continue
            else:
                train_data = np.row_stack((train_data, sample[j]))
    train_labels = train_data[:, 0].reshape(size, 1)  # 标签
    train_data = np.delete(train_data, [0], axis=1)  # 删除第一列
    # np.savetxt("train_data50_original.txt", train_data)  # 存储归一化之前的原始的真实样本，用于计算其中的最小值与最大值从而对训练集进行归一化
    # Normalized processing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_data1 = min_max_scaler.fit_transform(train_data)
    # print("hello", train_data1)
    train_data = MaxMinNormalization(train_data, min, max)
    return train_data, train_labels


# 选择高质量生成样本 训练数据：不同组生成样本；测试数据：真实样本；评价标准：真实样本中”异常样本“个数
def choose():
    # Load generated data
    train_data = np.loadtxt('CWGAN_data' + str(select_number) + '_level' + str(level_num) +'.txt')  # 生成数据1500*7=105000个
    test_data = np.loadtxt("train_data"+str(select_number)+"_level"+str(level_num)+"_normalization.txt")  # 真实数据
    train_data_ = np.split(train_data, 7)
    test_data_ = np.vsplit(test_data, 7)
    filter_n = 10  # 每一类筛选的次数
    for i in range(7):  # 7类，每类均需要训练一个VAE做异常检测
        result_temp_all = []  # 对真实样本“异常检测”检测出来的个数
        for j in range(filter_n):  # 每一类筛选10次 #可修改为筛选个数
            print("Class:%d/7  Filter_n:%d/%d" % (i+1, j+1, filter_n))
            num = random.sample(range(0, 1500), 1000)  # 生成数据每类1500个样本中选择1000个样本的下标
            train_data_temp = train_data_[i][num]  # 生成数据每类1500个样本中选择1000个样本 二维数组
            test_data_temp = test_data_[i]  # 选择该类对应的真实样本
            result_temp = mlp_vae_predict(train_data_temp, test_data_temp)  # 对真实样本进行“异常检测”
            if j == 0:
                train_data_temp_all = train_data_temp  # 临时选择的生成样本
            else:
                train_data_temp_all = np.row_stack((train_data_temp_all, train_data_temp))  # 存储临时选择的生成样本
            result_temp_all.append(result_temp)  # 存储对真实样本“异常检测”检测出来的个数
            print("temp anomaly numbers:", result_temp_all, '\n')
        min_index = result_temp_all.index(min(result_temp_all))  # 得到10组中最少异常数目的那组下标
        train_data_temp_all = np.reshape(train_data_temp_all, (filter_n, 1000, 65))  # 10组的生成样本
        #  存储最终的1000*7=7000组高质量生成样本
        if i == 0:
            train_data_select = train_data_temp_all[min_index]
        else:
            train_data_select = np.row_stack((train_data_select, train_data_temp_all[min_index]))
        print("目前筛选维度累计：", train_data_select.shape, "目标维度：（7000， 65）")
    # name = "select_generated_data.txt"
    # np.savetxt(name, train_data_select)
    np.savetxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_vae_od.txt", train_data_select)
    return train_data_select


# 分类阶段，训练数据：高质量生成样本， 测试数据： 真实数据
def classify(x_train1, y_train1, iteration):


    # x_test = np.loadtxt('test_data.txt')  # 真实故障样本
    # y_test = np.loadtxt('test_labels.txt')
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_test = min_max_scaler.fit_transform(x_test)

    min, max = get_min_max()
    # x_test, y_test = load_data(select_number, min, max)
    # 2020.4.17
    x_test, y_test = load_data(500, min, max)
    # Random Forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(x_train1, y_train1)
    RF_pre = rfc1.predict(x_test)
    RF_AC = accuracy_score(y_test, RF_pre)
    RF_f1 = f1_score(y_test, RF_pre, average='macro')

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
    # Main.py按照original.py, Ensemble.py, vae_od.py顺序执行，结果依次存入下面文件
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
    # 因为choose()使用的训练数据（真实数据）和测试数据（生成数据）均相同，那么训练出来的VAE参数相同,那么挑选出来的生成样本大致相同
    x_train1 = choose()
    # 创建对应的y label
    y_train1 = np.ones((1000, 1))
    init = 1
    for i in range(6):
        temp = np.ones((1000, 1)) + init
        y_train1 = np.row_stack((y_train1, temp))
        init += 1
    y_train1 = y_train1.ravel()
    # print(y_train1)
    for i in range(1):
        RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC, RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1\
            = classify(x_train1, y_train1, i)
        # print(type(RF_AC))返回的类型是np.float64,可使用append直接添加新数值
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
    # print("max_acc:", max(ave_RF), max(ave_SVM), max(ave_DT), max(ave_NB), max(ave_MLP), max(ave_KNN), max(ave_LG))
    # print("min_acc:", min(ave_RF), min(ave_SVM), min(ave_DT), min(ave_NB), min(ave_MLP), min(ave_KNN), min(ave_LG))
    # print("max_f1:", max(ave_RF_f1), max(ave_SVM_f1), max(ave_DT_f1), max(ave_NB_f1), max(ave_MLP_f1), max(ave_KNN_f1),
    #       max(ave_LG_f1))
    # print("min_f1:", min(ave_RF_f1), min(ave_SVM_f1), min(ave_DT_f1), min(ave_NB_f1), min(ave_MLP_f1), min(ave_KNN_f1),
    #       min(ave_LG_f1))
    # print("======Average accuracy and f1_score=======")
    # print("Random forest accuracy:" + str(Get_Average(ave_RF) * 100))
    # print("Random forest f1_score:" + str(Get_Average(ave_RF_f1) * 100))
    # print("=============")
    # print("SVM accuracy:" + str(Get_Average(ave_SVM) * 100))
    # print("SVM f1_score:" + str(Get_Average(ave_SVM_f1) * 100))
    # print("=============")
    # print("Decision tree accuracy:" + str(Get_Average(ave_DT) * 100))
    # print("Decision tree f1_score:" + str(Get_Average(ave_DT_f1) * 100))
    # print("=============")
    # print("Naive Bayes accuracy:" + str(Get_Average(ave_NB) * 100))
    # print("Naive Bayes f1_score:" + str(Get_Average(ave_NB_f1) * 100))
    # print("=============")
    # print("Multilayer perceptron accuracy:" + str(Get_Average(ave_MLP) * 100))
    # print("Multilayer perceptron f1_score:" + str(Get_Average(ave_MLP_f1) * 100))
    # print("=============")
    # print("KNN accuracy:" + str(Get_Average(ave_KNN) * 100))
    # print("KNN f1_score:" + str(Get_Average(ave_KNN_f1) * 100))
    # print("=============")
    # print("LogisticRegression accuracy:" + str(Get_Average(ave_LG) * 100))
    # print("LogisticRegression f1_score:" + str(Get_Average(ave_LG_f1) * 100))


# if __name__ == '__main__':
evaluation()

