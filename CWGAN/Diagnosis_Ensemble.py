# 使用集成学习方法筛选高质量生成样本
# 重新选择真实样本测试集，并使用原本真实样本训练集的最小值和最大值做归一化

# -*- coding: utf-8 -*-
"""
Created on  June  5 10:54:44 2019

@author: Jianye Su
"""

import numpy as np
import scipy.io as sio
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
level_num = 3  # 等级
select_number = 50

# Load generated data 生成数据在筛选过程作为测试数据，在分类过程作为训练数据
test_data = np.loadtxt('CWGAN_data' + str(select_number) + '_level' + str(level_num) +'.txt')  # 生成数据
test_labels = np.loadtxt('labels1500.txt')

# Load Real data 训练GAN时的真实数据在筛选过程作为训练数据，分类过程的测试数据重新选择真实数据
train_data = np.loadtxt('train_data'+str(select_number)+"_level"+str(level_num)+'_normalization.txt')
train_labels = np.loadtxt('train_labels'+str(select_number)+'.txt')


#  获取训练集的最小值和最大值，测试集的min-max归一化使用训练集的最小值与最大值
def get_min_max():
    train_data = np.loadtxt('train_data'+str(select_number)+"_level"+str(level_num)+'_original.txt')
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
    sample = TR_sample_temp['num']  # 每类5191个，故数据总量为36337*66，最后一列为标

    # Select sample
    # 从每个区间中（共7个区间，每个区间5191个样本）随机获取t个（50个）元素，作为一个片断返回（共350个）
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
    train_data = MaxMinNormalization(train_data, min, max)
    return train_data, train_labels


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


# 投票评估模型，通过分类器，模型训练数据：真实故障样本， 测试数据：生成故障样本
def training2(test_sample):
    # SVM

    # print("Phase 1 SVM parameters selecting...")
    # parameters = {
    #     'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #     'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #     'kernel': ['rbf'],
    #     'decision_function_shape': ['ovr']
    # }
    # svc = svm.SVC()
    # # 网格搜索设置模型和评价指标，开始用不同的参数训练模型
    # grid_search = GridSearchCV(svc, parameters, scoring='accuracy', cv=10)
    # # 十折交叉验证,将训练集分成十份，轮流将其中9份作为训练数据，1份作为验证数据，进行试验。每次试验都会得出相应的正确率（或差错率）,一般还需要进行多次10折交叉验证（例如10次10折交叉验证），再求其均值，作为对算法准确性的估计。
    # grid_search.fit(train_data, train_labels.ravel())  # ravel（）将多维数组降为一维
    # # 最佳参数输出
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    # # 这里使用之前得到的最佳参数对模型进行重新训练，在训练时，就可以将所有的数据都作为训练数据全部投入到模型中去，这样就不会浪费个别数据了
    # clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True).fit(train_data, train_labels.ravel())

    clf = svm.SVC(kernel='rbf', C=19, gamma=0.1)
    clf.set_params(kernel='rbf', probability=True).fit(train_data, train_labels)
    SVM_predict = clf.predict(test_sample)

    # Random Forest
    rfc1 = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_data, train_labels)
    RF_predict = rfc1.predict(test_sample)

    # knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_labels)
    knn_predict = knn.predict(test_sample)
    return SVM_predict, RF_predict, knn_predict


# 第一次筛选
# Vote进行投票选择高质量生成样本，通过3种模型预测选择票数多的样本
def VOTE(test_sample, test_label):
    test_label = test_label.reshape(-1, 1)
    SVM_pre, RF_pre, knn_pre = training2(test_sample)
    Votenum = np.array([])
    pre_labels = np.array([])
    delet1 = np.array([])

    # 投票标签
    for i in range(test_label.shape[0]):
        Vote = 0
        lab = 0  # 投票选出来的标签
        if (SVM_pre[i] == RF_pre[i]) & (SVM_pre[i] == knn_pre[i]):
            Vote = 3
            lab = SVM_pre[i]
        if (SVM_pre[i] == RF_pre[i]) & (SVM_pre[i] != knn_pre[i]):
            Vote = 2
            lab = SVM_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (SVM_pre[i] == knn_pre[i]):
            Vote = 2
            lab = SVM_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (RF_pre[i] == knn_pre[i]):
            Vote = 2
            lab = RF_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (SVM_pre[i] != knn_pre[i]) & (knn_pre[i] != RF_pre[i]):
            Vote = 0
            lab = RF_pre[i]

        pre_labels = np.append(pre_labels, lab)  # 每个生成样本对应的预测标签
        Votenum = np.append(Votenum, Vote)  # 每个生成样本预测标签的票数

    # 预挑选票数大于2的生成样本，后期再把预测标签与自带标签进行比对
    for j in range(test_label.shape[0]):
        if Votenum[j] < 2:
            delet1 = np.append(delet1, j)  # 获取下标
            delet1 = delet1.astype('int64')

    # 预选择出来的样本
    init_choose_data = np.delete(test_sample, delet1, axis=0)
    init_choose_label = np.delete(test_label, delet1, axis=0)
    # 删除的样本
    del_label = test_label[delet1, :]
    del_sample = test_sample[delet1, :]
    # 预选择出来的样本的预测标签
    pre_labels = np.delete(pre_labels, delet1, axis=0)
    lens = pre_labels.shape[0]
    return init_choose_data, init_choose_label, pre_labels, lens, del_label, del_sample


# 第二次筛选高质量生成故障样本
# 预测标签与自带标签比对
def CHOSE():
    delet2 = np.array([])
    init_choose_data, init_choose_label, label, lens, del_label, del_sample = VOTE(test_data, test_labels)
    # 第二次筛选
    for i in range(lens):
        if label[i] != init_choose_label[i]:
            delet2 = np.append(delet2, i).astype('int32')  # 返回删除下标
            delet2 = delet2.astype('int32')

    # 选择出的高质量样本
    choose_data = np.delete(init_choose_data, delet2, axis=0)
    choose_label = np.delete(init_choose_label, delet2, axis=0)
    choose_data_all = choose_data
    choose_label_all = choose_label

    # 筛选出来的高质量样本数量不够，对已删除样本重新筛选。
    while choose_label.shape[0] <= 7000:
        del_label = np.append(del_label, init_choose_label[delet2, :])  # 两次删除的样本标签
        del_sample = np.row_stack((del_sample, init_choose_data[delet2, :]))  # 两次删除的样本
        init_choose_data, init_choose_label, label, lens, del_label, del_sample = VOTE(del_sample,
                                                                                       del_label)  # 对删除过的标签重新选择防止误分类
        delet2 = np.array([])
        for i in range(lens):
            if label[i] != init_choose_label[i]:
                delet2 = np.append(delet2, i)  # 返回删除下标
                delet2 = delet2.astype('int32')
        choose_data = np.delete(init_choose_data, delet2, axis=0)
        choose_label = np.delete(init_choose_label, delet2, axis=0)

        choose_data_all = np.row_stack((choose_data_all, choose_data))
        choose_label_all = np.append(choose_label_all, choose_label)

        if choose_label.shape[0] == 0:
            break
    np.savetxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_Ensemble.txt", choose_data_all)
    return choose_data_all, choose_label_all


# 对评估出的高质量样本进行分类，分类器训练数据：生成故障样本， 测试数据：真实故障样本
def classify(x_train1, y_train1, i):
    # print(x_train1.shape)
    y_train1 = y_train1.ravel()
    # print(y_train1.shape)


    # x_test = np.loadtxt('train_data50.txt')  # 真实故障样本作为测试数据
    # y_test = np.loadtxt('train_labels50.txt')
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_test = min_max_scaler.fit_transform(x_test)

    min, max = get_min_max()
    x_test, y_test = load_data(500, min, max) # 重新加载其他真实数据测试，使用原本真实数据的最小值与最大值归一化


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
    # # 网格搜索设置模型和评价指标，开始用不同的参数训练模型
    # grid_search = GridSearchCV(svc, parameters, scoring='accuracy', cv=5)
    # # 十折交叉验证,将训练集分成十份，轮流将其中9份作为训练数据，1份作为验证数据，进行试验。每次试验都会得出相应的正确率（或差错率）,一般还需要进行多次10折交叉验证（例如10次10折交叉验证），再求其均值，作为对算法准确性的估计。
    # grid_search.fit(x_train1, y_train1.ravel())  # ravel（）将多维数组降为一维
    # # 最佳参数输出
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print("hello:", para, val)
    # # 这里使用之前得到的最佳参数对模型进行重新训练，在训练时，就可以将所有的数据都作为训练数据全部投入到模型中去，这样就不会浪费个别数据了
    # clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True).fit(
    #     x_train1, y_train1.ravel())

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
    print("===== Diagnosis Ensemble evaluation %d/1=======" % (i+1))
    print('Ensemble Accuracy:')
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


# if __name__ == "__main__":
x_train1, y_train1 = CHOSE()  # 分类过程选出高质量生成数据作为训练数据
for i in range(1):
    RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC, RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1 \
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
# print("max_f1:", max(ave_RF_f1), max(ave_SVM_f1), max(ave_DT_f1), max(ave_NB_f1), max(ave_MLP_f1), max(ave_KNN_f1), max(ave_LG_f1))
# print("min_f1:", min(ave_RF_f1), min(ave_SVM_f1), min(ave_DT_f1), min(ave_NB_f1), min(ave_MLP_f1), min(ave_KNN_f1), min(ave_LG_f1))
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
