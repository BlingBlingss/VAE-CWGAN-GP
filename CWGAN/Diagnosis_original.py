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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from option import Options


op = Options()
level_num = op.level_num
select_number = op.select_number
min_max_scaler = preprocessing.MinMaxScaler()


def load_data(select_num, name1, name2):
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
    train_labels = train_data[:, 0]
    train_data = np.delete(train_data, [0], axis=1)
    train_data = min_max_scaler.fit_transform(train_data)
    np.savetxt(name1, train_data)
    np.savetxt(name2, train_labels)
    return train_data, train_labels


true_fault_data_train = np.loadtxt("train_data"+str(select_number)+"_level"+str(level_num)+"_normalization.txt")
true_fault_label_train = np.loadtxt("train_labels"+str(select_number)+".txt")
true_fault_data_test, true_fault_label_test = load_data(500,
                                                        name1="./data_results/RealFaultdataLevel"+str(level_num)+"total-3500.txt", name2 = "./data_results/RealFaultlabeltotal-3500.txt")
train_dataAndlabel = np.column_stack((true_fault_label_train, true_fault_data_train))
np.random.shuffle(train_dataAndlabel)
train_data = train_dataAndlabel[:, 1:]
train_label = train_dataAndlabel[:, 0]
test_data = true_fault_data_test
test_label = true_fault_label_test



def classify(train_data, train_label):
    train_label = train_label.ravel()

    # Random Forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_data, train_label)
    RF_pre = rfc1.predict(test_data)
    RF_AC = accuracy_score(test_label, RF_pre)
    RF_f1 = f1_score(test_label, RF_pre, average='macro')

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
    clf.set_params(kernel='rbf', probability=True).fit(train_data, train_label)
    clf.predict(train_data)
    test_pre = clf.predict(test_data)
    SVM_AC = accuracy_score(test_label, test_pre)
    SVM_f1 = f1_score(test_label, test_pre, average='macro')

    # decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_data, train_label)
    dt_pre = dtc.predict(test_data)
    DT_AC = accuracy_score(test_label, dt_pre)
    DT_f1 = f1_score(test_label, dt_pre, average='macro')

    # Bayes
    mnb = MultinomialNB()
    mnb.fit(train_data, train_label)
    NB_predict = mnb.predict(test_data)
    NB_AC = accuracy_score(test_label, NB_predict)
    NB_f1 = f1_score(test_label, NB_predict, average='macro')

    # Multilayer perceptron
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_data, train_label)
    MLP_predict = MLP.predict(test_data)
    MLP_AC = accuracy_score(test_label, MLP_predict)
    MLP_f1 = f1_score(test_label, MLP_predict, average='macro')

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_label)
    knn_predict = knn.predict(test_data)
    KNN_AC = accuracy_score(test_label, knn_predict)
    KNN_f1 = f1_score(test_label, knn_predict, average='macro')

    # LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(train_data, train_label)
    lg_predict = classifier.predict(test_data)
    LG_AC = accuracy_score(test_label, lg_predict)
    LG_f1 = f1_score(test_label, lg_predict, average='macro')

    print("===== Diagnosis original=======")
    print('Original Accuracy:')
    print(RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC)
    print('F1-score')
    print(RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1)
    file_name1 = "./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_Accuracy_result.txt"
    file_name2 = "./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_f1_score_result.txt"
    with open(file_name1, "a") as f:
        f.writelines([str(RF_AC), ' ', str(SVM_AC), ' ', str(DT_AC), ' ', str(NB_AC), ' ', str(MLP_AC), ' ', str(KNN_AC), ' ', str(LG_AC), '\n'])
    with open(file_name2, "a") as f:
        f.writelines([str(RF_f1), ' ', str(SVM_f1), ' ', str(DT_f1), ' ', str(NB_f1), ' ', str(MLP_f1), ' ', str(KNN_f1), ' ', str(LG_f1), '\n'])
    return RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC


RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC = classify(train_data, train_label)

