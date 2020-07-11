# 使用GAN扩充故障训练样本，正常样本与筛选过的生成故障样本做检测

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

level_num = 3
select_number = 50
min_max_scaler = preprocessing.MinMaxScaler()
# 高质量故障样本（由归一化的真实样本生成，故不再需要归一化）
# gen_fault_data = np.loadtxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_Ensemble.txt")  # 使用挑选过的高质量生成数据
gen_fault_data = np.loadtxt(".\data_results\CWGAN_data"+str(select_number)+"_Level"+str(level_num)+"SelectedData_vae_od.txt")  # 7000
true_fault_data = np.loadtxt("./data_results/RealFaultdataLevel"+str(level_num)+"total-3500.txt")  # 3500
# gen_fault_data = min_max_scaler.fit_transform(gen_fault_data)
gen_fault_label = np.ones(gen_fault_data.shape[0])  # 所有生成故障均标记为标签1 共1000*7=7000个
true_fault_label = np.ones(true_fault_data.shape[0])  # 所有真实故障均标记为标签1 共select_num*7个
# gen_fault_data = np.column_stack((ones, gen_fault_data))

# 正常样本 （需要归一化）
Normal_mat = sio.loadmat('Normal.mat')  # 正常数据15573个
Normal_dataAndlabel = Normal_mat['num']
Normal_label = Normal_dataAndlabel[:, 0]
Normal_data = Normal_dataAndlabel[:, 1:]
Normal_data = min_max_scaler.fit_transform(Normal_data)
# 训练集：12000(正常)+7000（生成的故障） 测试集：3573（正常）+3500(真实的故障，只不过全标记为1)
Normal_data_train = Normal_data[:12000, :]
Normal_label_train = Normal_label[:12000]
Normal_data_test = Normal_data[12000:, :]
Normal_label_test = Normal_label[12000:]

# Train 训练集：12000(正常)+7000（生成的故障）
train_data = np.row_stack((Normal_data_train, gen_fault_data))
# 一维数组不能用np.row_stack,使用append
train_label = np.append(Normal_label_train, gen_fault_label)
train_dataAndlabel = np.column_stack((train_label, train_data))
# 打乱
np.random.shuffle(train_dataAndlabel)
train_data = train_dataAndlabel[:, 1:]
train_label = train_dataAndlabel[:, 0]

# Test 测试集：3573（正常）+3500(真实的故障，只不过全标记为1)
test_data = np.row_stack((Normal_data_test, true_fault_data))
test_label = np.append(Normal_label_test, true_fault_label)
# test_dataAndlabel = np.column_stack((test_label, test_data))
# np.random.shuffle(test_dataAndlabel)
# test_data = test_dataAndlabel[:, 1:]
# test_label = test_dataAndlabel[:, 0]

# 对评估出的高质量样本进行分类，分类器训练数据：生成故障样本， 测试数据：真实故障样本
def classify(train_data, train_label):
    # print(train_data.shape)
    train_label = train_label.ravel()
    # print(train_label.shape)

    # Random Forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_data, train_label)
    RF_pre = rfc1.predict(test_data)
    RF_AC = accuracy_score(test_label, RF_pre)
    RF_PRECISION = precision_score(test_label, RF_pre, average='binary')
    RF_RECALL = recall_score(test_label, RF_pre, average='binary')
    RF_f1 = f1_score(test_label, RF_pre, average='binary')
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
    # grid_search.fit(train_data, train_label.ravel())  # ravel（）将多维数组降为一维
    # # 最佳参数输出
    # best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print("hello:", para, val)
    # # 这里使用之前得到的最佳参数对模型进行重新训练，在训练时，就可以将所有的数据都作为训练数据全部投入到模型中去，这样就不会浪费个别数据了
    # clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True).fit(
    #     train_data, train_label.ravel())

    clf = SVC(kernel='rbf', C=9, gamma=0.1)
    clf.set_params(kernel='rbf', probability=True).fit(train_data, train_label)
    clf.predict(train_data)
    test_pre = clf.predict(test_data)
    SVM_AC = accuracy_score(test_label, test_pre)
    SVM_f1 = f1_score(test_label, test_pre, average='binary')

    # decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_data, train_label)
    dt_pre = dtc.predict(test_data)
    DT_AC = accuracy_score(test_label, dt_pre)
    DT_f1 = f1_score(test_label, dt_pre, average='binary')

    # Bayes
    mnb = MultinomialNB()
    mnb.fit(train_data, train_label)
    NB_predict = mnb.predict(test_data)
    NB_AC = accuracy_score(test_label, NB_predict)
    NB_f1 = f1_score(test_label, NB_predict, average='binary')

    # Multilayer perceptron
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_data, train_label)
    MLP_predict = MLP.predict(test_data)
    MLP_AC = accuracy_score(test_label, MLP_predict)
    MLP_f1 = f1_score(test_label, MLP_predict, average='binary')

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_label)
    knn_predict = knn.predict(test_data)
    KNN_AC = accuracy_score(test_label, knn_predict)
    KNN_f1 = f1_score(test_label, knn_predict, average='binary')

    # LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(train_data, train_label)
    lg_predict = classifier.predict(test_data)
    LG_AC = accuracy_score(test_label, lg_predict)
    LG_f1 = f1_score(test_label, lg_predict, average='binary')

    print("===== Detection vae_od evaluation=======")
    print('vae_od Accuracy:')
    print(RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC)
    print('F1-score')
    print(RF_f1, SVM_f1, DT_f1, NB_f1, MLP_f1, KNN_f1, LG_f1)
    # Main.py按照original.py, Ensemble.py, vae_od.py顺序执行，结果依次存入下面文件
    file_name1 = "./temp_result/Detection_" + str(select_number) + "Level" + str(level_num) + "_Accuracy_result.txt"
    file_name2 = "./temp_result/Detection_" + str(select_number) + "Level" + str(level_num) + "_f1_score_result.txt"
    with open(file_name1, "a") as f:
        f.writelines(
            [str(RF_AC), ' ', str(SVM_AC), ' ', str(DT_AC), ' ', str(NB_AC), ' ', str(MLP_AC), ' ', str(KNN_AC), ' ',
             str(LG_AC), '\n'])
    with open(file_name2, "a") as f:
        f.writelines(
            [str(RF_f1), ' ', str(SVM_f1), ' ', str(DT_f1), ' ', str(NB_f1), ' ', str(MLP_f1), ' ', str(KNN_f1), ' ',
             str(LG_f1), '\n'])
    return RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC


# if __name__ == "__main__":
RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC = classify(train_data, train_label)
# print("=============")
# print("Random forest accuracy:" + str(RF_AC))
# print("=============")
# print("SVM accuracy:" + str(SVM_AC))
# print("=============")
# print("Decision tree accuracy:" + str(DT_AC))
# print("=============")
# print("Naive Bayes accuracy:" + str(NB_AC))
# print("=============")
# print("Multilayer perceptron accuracy:" + str(MLP_AC))
# print("=============")
# print("KNN accuracy:" + str(KNN_AC))
# print("=============")
# print("LogisticRegression accuracy:" + str(LG_AC))