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
from option import Options


op = Options()
level_num = op.level_num
select_number = op.select_number
test_data = np.loadtxt('CWGAN_data' + str(select_number) + '_level' + str(level_num) +'.txt')
test_labels = np.loadtxt('labels1500.txt')


train_data = np.loadtxt('train_data'+str(select_number)+"_level"+str(level_num)+'_normalization.txt')
train_labels = np.loadtxt('train_labels'+str(select_number)+'.txt')


def get_min_max():
    train_data = np.loadtxt('train_data'+str(select_number)+"_level"+str(level_num)+'_original.txt')
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


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


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
    # grid_search = GridSearchCV(svc, parameters, scoring='accuracy', cv=10)
    # grid_search.fit(train_data, train_labels.ravel())
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
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


def VOTE(test_sample, test_label):
    test_label = test_label.reshape(-1, 1)
    SVM_pre, RF_pre, knn_pre = training2(test_sample)
    Votenum = np.array([])
    pre_labels = np.array([])
    delet1 = np.array([])


    for i in range(test_label.shape[0]):
        Vote = 0
        lab = 0
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

        pre_labels = np.append(pre_labels, lab)
        Votenum = np.append(Votenum, Vote)

    for j in range(test_label.shape[0]):
        if Votenum[j] < 2:
            delet1 = np.append(delet1, j)
            delet1 = delet1.astype('int64')

    init_choose_data = np.delete(test_sample, delet1, axis=0)
    init_choose_label = np.delete(test_label, delet1, axis=0)
    del_label = test_label[delet1, :]
    del_sample = test_sample[delet1, :]
    pre_labels = np.delete(pre_labels, delet1, axis=0)
    lens = pre_labels.shape[0]
    return init_choose_data, init_choose_label, pre_labels, lens, del_label, del_sample


def CHOSE():
    delet2 = np.array([])
    init_choose_data, init_choose_label, label, lens, del_label, del_sample = VOTE(test_data, test_labels)
    for i in range(lens):
        if label[i] != init_choose_label[i]:
            delet2 = np.append(delet2, i).astype('int32')
            delet2 = delet2.astype('int32')
    choose_data = np.delete(init_choose_data, delet2, axis=0)
    choose_label = np.delete(init_choose_label, delet2, axis=0)
    choose_data_all = choose_data
    choose_label_all = choose_label
    while choose_label.shape[0] <= 7000:
        del_label = np.append(del_label, init_choose_label[delet2, :])
        del_sample = np.row_stack((del_sample, init_choose_data[delet2, :]))
        init_choose_data, init_choose_label, label, lens, del_label, del_sample = VOTE(del_sample,
                                                                                       del_label)
        delet2 = np.array([])
        for i in range(lens):
            if label[i] != init_choose_label[i]:
                delet2 = np.append(delet2, i)
                delet2 = delet2.astype('int32')
        choose_data = np.delete(init_choose_data, delet2, axis=0)
        choose_label = np.delete(init_choose_label, delet2, axis=0)
        choose_data_all = np.row_stack((choose_data_all, choose_data))
        choose_label_all = np.append(choose_label_all, choose_label)
        if choose_label.shape[0] == 0:
            break
    np.savetxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_Ensemble.txt", choose_data_all)
    return choose_data_all, choose_label_all


def classify(x_train1, y_train1, i):
    y_train1 = y_train1.ravel()
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
    # grid_search.fit(x_train1, y_train1.ravel())
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print("hello:", para, val)
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


x_train1, y_train1 = CHOSE()
for i in range(1):
    RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC, RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1 \
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
