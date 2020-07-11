#  添加残差网络

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:54:44 2018

@author: Zhongchaowen
"""

import os, time, pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import scipy.io as sio
import tensorflow.contrib.slim as slim

tf.reset_default_graph()
# training parameters
select_number = 50  # 选择每类选取的故障样本数
M_size = select_number * 5  # 5类真实故障样本共250个
N_size = 10  # 选取的特征数

LabN_size = 5  # onehot表示标签向量

G_size = select_number * 5  # 生成故障样本数250个
Zn_size = 100  # 噪声维度

lr_g = 0.0001
lr_D = 0.0001
train_epoch = 20000  # 20000

train_hist = dict()  # train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# variables : input
gy = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5


# 加载真实故障数据
def load_data(select_num, name1="train_data50_normalization" + ".txt", name2="train_labels50_normalization" + ".txt"):  # select_num = 5 #选择每类选取的故障样本数

    size = select_num * 5   # 共5类
    total_num = 6000
    step_size = 1200
    TR_sample_temp = sio.loadmat('b.mat')
    data = TR_sample_temp['b']  # 除第二类为1500个外，剩余四类每类1200个，故数据总量为6300*11，最后一列为标签
    # feature select
    chosen = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 选择的特征列数，理应10个特征，但是最后一列是类别标签，所以选择11个
    L = len(chosen)
    for i in range(L):
        if i == 0:
            temp1 = data[:, chosen[i]]  # 选出每个样本的选择的特征列
            sample = temp1
        else:
            temp1 = data[:, chosen[i]]
            sample = np.column_stack((sample, temp1))  # 变为样本表格，每行代表一个样本，每列代表一个特征
    # 随机删除300条 返回一个下表列表
    Del = random.sample(range(2400, 3899), 300)
    sample = np.delete(sample, Del, axis=0)

    # Select sample
    # 从每个区间中（共5个区间，每个区间1200个样本）随机获取t个（50个）元素，作为一个片断返回（共250个）
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
    train_labels = train_data[:, 10].reshape(size, 1)  # 标签
    train_data = np.delete(train_data, [10], axis=1)  # 删除最后一列
    np.savetxt("train_data50_original.txt", train_data)  # 存储归一化之前的原始的真实样本，用于计算其中的最小值与最大值从而对训练集进行归一化
    # Normalized processing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_data1 = min_max_scaler.fit_transform(train_data)
    # print("hello", train_data1)
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)
    train_data = MaxMinNormalization(train_data, min, max)
    np.savetxt(name1, train_data)  # 存储归一化之后的原始的真实样本
    np.savetxt(name2, train_labels)
    return train_data, train_labels


def MaxMinNormalization(x, min, max):
    x = (x - min) / (max - min)
    return x


# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)  # 0.6
    f2 = 0.5 * (1 - leak)  # 0.4
    return f1 * X + f2 * tf.abs(X)  # 即f(x) = 0.6x + 0.4|x|


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

'''===Gnerator Parameters==='''
#  原始层数
# z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
# y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5个one-hot表示的标签
# G_w1 = tf.Variable(xavier_init([Zn_size+LabN_size, 128]))
# G_b1 = tf.Variable(tf.zeros([128]))
# G_w2 = tf.Variable(xavier_init([128, 64]))
# G_b2 = tf.Variable(tf.zeros([64]))
# G_w3 = tf.Variable(xavier_init([64, 32]))
# G_b3 = tf.Variable(tf.zeros([32]))
# G_w4 = tf.Variable(xavier_init([32, 10]))
# G_b4 = tf.Variable(tf.zeros([10]))
# G_vars = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_w4, G_b4]


#  增加一层，生成器为5层
# 0.924 0.936 0.884 0.364 0.2 0.908 0.764
# z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
# y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5个one-hot表示的标签
# G_w1 = tf.Variable(xavier_init([Zn_size+LabN_size, 105]))
# G_b1 = tf.Variable(tf.zeros([105]))
# G_w2 = tf.Variable(xavier_init([105, 105]))
# G_b2 = tf.Variable(tf.zeros([105]))
# G_w3 = tf.Variable(xavier_init([105, 105]))
# G_b3 = tf.Variable(tf.zeros([105]))
# G_w4 = tf.Variable(xavier_init([105, 105]))
# G_b4 = tf.Variable(tf.zeros([105]))
# G_w5 = tf.Variable(xavier_init([105, 10]))
# G_b5 = tf.Variable(tf.zeros([10]))
# G_vars = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_w4, G_b4, G_w5, G_b5]


#  接着增加层数,生成器为7层
# z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
# y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5个one-hot表示的标签
# G_w1 = tf.Variable(xavier_init([Zn_size+LabN_size, 128]))
# G_b1 = tf.Variable(tf.zeros([128]))
# G_w2 = tf.Variable(xavier_init([128, 64]))
# G_b2 = tf.Variable(tf.zeros([64]))
# G_w3 = tf.Variable(xavier_init([64, 64]))
# G_b3 = tf.Variable(tf.zeros([64]))
# G_w4 = tf.Variable(xavier_init([64, 64]))
# G_b4 = tf.Variable(tf.zeros([64]))
# G_w5 = tf.Variable(xavier_init([64, 64]))
# G_b5 = tf.Variable(tf.zeros([64]))
# G_w6 = tf.Variable(xavier_init([64, 32]))
# G_b6 = tf.Variable(tf.zeros([32]))
# G_w7 = tf.Variable(xavier_init([32, 10]))
# G_b7 = tf.Variable(tf.zeros([10]))
# G_vars = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_w4, G_b4, G_w5, G_b5, G_w6, G_b6, G_w7, G_b7]


#  生成器7层, 搭建Resnet
z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5个one-hot表示的标签
G_w1 = tf.Variable(xavier_init([Zn_size+LabN_size, 105]))
G_b1 = tf.Variable(tf.zeros([105]))
G_w2 = tf.Variable(xavier_init([105, 105]))
G_b2 = tf.Variable(tf.zeros([105]))
G_w3 = tf.Variable(xavier_init([105, 105]))
G_b3 = tf.Variable(tf.zeros([105]))
G_w4 = tf.Variable(xavier_init([105, 105]))
G_b4 = tf.Variable(tf.zeros([105]))
G_w5 = tf.Variable(xavier_init([105, 105]))
G_b5 = tf.Variable(tf.zeros([105]))
G_w6 = tf.Variable(xavier_init([105, 105]))
G_b6 = tf.Variable(tf.zeros([105]))
G_w7 = tf.Variable(xavier_init([105, 10]))
G_b7 = tf.Variable(tf.zeros([10]))
G_vars = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_w4, G_b4, G_w5, G_b5, G_w6, G_b6, G_w7, G_b7]


# z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
# y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 5个one-hot表示的标签
# G_w1 = tf.Variable(xavier_init([Zn_size+LabN_size, 105]))
# G_b1 = tf.Variable(tf.zeros([105]))
# G_w2 = tf.Variable(xavier_init([105, 105]))
# G_b2 = tf.Variable(tf.zeros([105]))
# G_w3 = tf.Variable(xavier_init([105, 105]))
# G_b3 = tf.Variable(tf.zeros([105]))
# G_w4 = tf.Variable(xavier_init([105, 105]))
# G_b4 = tf.Variable(tf.zeros([105]))
# G_w5 = tf.Variable(xavier_init([105, 10]))
# G_b5 = tf.Variable(tf.zeros([10]))
# G_vars = [G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_w4, G_b4, G_w5, G_b5]

'''===Discriminator Parameters==='''
x = tf.placeholder(tf.float32, shape=(None, N_size))  # 10个特征
D_w1 = tf.Variable(xavier_init([N_size+LabN_size, 64]))
D_b1 = tf.Variable(tf.zeros([64]))
D_w2 = tf.Variable(xavier_init([64, 128]))
D_b2 = tf.Variable(tf.zeros([128]))
D_w3 = tf.Variable(xavier_init([128, 64]))
D_b3 = tf.Variable(tf.zeros([64]))
D_w4 = tf.Variable(xavier_init([64, 32]))
D_b4 = tf.Variable(tf.zeros([32]))
D_w5 = tf.Variable(xavier_init([32, 1]))
D_b5 = tf.Variable(tf.zeros([1]))
D_vars = [D_w1, D_w2, D_w3, D_w4, D_w5, D_b1, D_b2, D_b3, D_b4, D_b5]


# G(z) 原始层数
# def generator(x, y):
#     input = tf.concat(values=[x, y], axis=1)
#     a1 = tf.nn.relu(tf.matmul(input, G_w1)+G_b1)
#     a2 = tf.nn.relu(tf.matmul(a1, G_w2)+G_b2)
#     a3 = tf.nn.relu(tf.matmul(a2, G_w3)+G_b3)
#     output = tf.nn.relu(tf.matmul(a3, G_w4)+G_b4)
#     return output

# 单纯增加层数 5层
# def generator(x, y):
#     input = tf.concat(values=[x, y], axis=1)
#     a1 = tf.nn.relu(tf.matmul(input, G_w1)+G_b1)
#     a2 = tf.nn.relu(tf.matmul(a1, G_w2)+G_b2)
#     a3 = tf.nn.relu(tf.matmul(a2, G_w3)+G_b3)
#     a4 = tf.nn.relu(tf.matmul(a3, G_w4)+G_b4)
#     output = tf.nn.relu(tf.matmul(a4, G_w5)+G_b5)
#     return output


# 增加层数 7层
# def generator(x, y):
#     input = tf.concat(values=[x, y], axis=1)
#     a1 = tf.nn.relu(tf.matmul(input, G_w1)+G_b1)
#     a2 = tf.nn.relu(tf.matmul(a1, G_w2)+G_b2)
#     a3 = tf.nn.relu(tf.matmul(a2, G_w3)+G_b3)
#     a4 = tf.nn.relu(tf.matmul(a3, G_w4)+G_b4)
#     a5 = tf.nn.relu(tf.matmul(a4, G_w5) + G_b5)
#     a6 = tf.nn.relu(tf.matmul(a5, G_w6) + G_b6)
#     output = tf.nn.relu(tf.matmul(a6, G_w7)+G_b7)
#     return output


# 插入的a1是在线性激活之后，非线性激活之前 5层resnet
# def generator(x, y):
#     input = tf.concat(values=[x, y], axis=1)
#     a1 = tf.nn.relu(tf.matmul(input, G_w1)+G_b1)
#     a2 = tf.nn.relu(tf.matmul(a1, G_w2)+G_b2+input)
#     a3 = tf.nn.relu(tf.matmul(a2, G_w3)+G_b3)
#     a4 = tf.nn.relu(tf.matmul(a3, G_w4)+G_b4+a2)
#     output = tf.nn.relu(tf.matmul(a4, G_w5)+G_b5)
#     return output

# 插入的a1是在线性激活之后，非线性激活之前 7层resnet
def generator(x, y):
    input = tf.concat(values=[x, y], axis=1)
    a1 = tf.nn.relu(tf.matmul(input, G_w1)+G_b1)
    a2 = tf.nn.relu(tf.matmul(a1, G_w2)+G_b2+input)
    a3 = tf.nn.relu(tf.matmul(a2, G_w3)+G_b3)
    a4 = tf.nn.relu(tf.matmul(a3, G_w4)+G_b4+a2)
    a5 = tf.nn.relu(tf.matmul(a4, G_w5)+G_b5)
    a6 = tf.nn.relu(tf.matmul(a5, G_w6)+G_b6+a4)
    output = tf.nn.relu(tf.matmul(a6, G_w7)+G_b7)
    return output


# D(x)
def discriminator(x, y):
    input = tf.concat(values=[x, y], axis=1)
    a1 = tf.nn.relu(tf.matmul(input, D_w1)+D_b1)
    a2 = tf.nn.relu(tf.matmul(a1, D_w2)+D_b2)
    a3 = tf.nn.relu(tf.matmul(a2, D_w3)+D_b3)
    a4 = tf.nn.relu(tf.matmul(a3, D_w4)+D_b4)
    output = tf.matmul(a4, D_w5)+D_b5
    return output

def one_hot(y, size):  # y为实际标签，size为样本个数
    label = []
    for i in range(size):  # size为样本个数
        a = int(y[i]) - 1
        temp = [0, 0, 0, 0, 0]  # 共5类故障
        temp[a] = 1
        label.extend(temp)  # 在列表末尾一次性追加另一个序列中的多个值
    label = np.array(label).reshape(size, 5)  # 最后重塑
    return label


# 返回one-hot表示的标签和对应的噪声
def G_labels(select_num, size, set_name=False):  # size为噪声维度

    t = 4
    # random.uniform(x, y, (m, n)) 方法将随机生成m*n维实数数组，它在 [x,y）范围内,且服从均匀分布。
    temp_z_ = np.random.uniform(-1, 1, (select_num, size))
    z_ = temp_z_
    fixed_y_ = np.ones((select_num, 1))  # 全为1
    j = 1
    for i in range(t):  # 生成每个类别的标签及其对应噪声
        temp = np.ones((select_num, 1)) + j  # 全为2或3或4或5
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)  # 生成标签 concatenate((a, b),axis=0)用于将数组a, b进行连接[[a], [b]]
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)  # 生成噪声 Matrix stitching [select_num*size]*6 6个循环
    y = one_hot(fixed_y_, select_num * 5)  # 6类
    if set_name:
        name = "labels1500" + ".txt"
        np.savetxt(name, fixed_y_)
    return y, z_  # 返回one-hot表示的标签和对应的噪声


def show_result(epoch_num):  # 网络训练好后生成最终的故障样本
    with tf.variable_scope('show_result'):
        if epoch_num == 19999:  # 19999
            G_y, fixed_z_ = G_labels(1500, 100, True)  # 返回顺序排列的生成标签和对应的噪声共9000个
            G = sess.run(G_z, {z: fixed_z_, gy: G_y})  # 返回含有11个特征的生成样本
            G_sample = G
            name = "CWGAN_data" + str(select_number) + ".txt"
            np.savetxt(name, G_sample)
            return G_sample


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)  # legend为标准曲线含义（根据曲线的label,这里为D_loss, G_loss）
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)  # 保存路径

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 1、定义计算图所有的计算
    # networks : generator
    G_z = generator(z, gy)  # 返回生成的含有11个特征的生成样本

    # networks : discriminator
    D_real_logits = discriminator(x, y)  # 原数据评分,使用默认的reuse=False先创建变量
    # D_fake_logits = discriminator(G_z, y, reuse=True)  # 生成数据评分，reuse = True复用变量
    D_fake_logits = discriminator(G_z, y)  # 生成数据评分，reuse = True复用变量

    # 计算损失函数
    # Wgan trick 加入惩罚项
    eps = tf.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)  # 返回一个矩阵，产生于minval和maxval之间，产生的值是均匀分布的。G_size为30
    X_inter = eps * x + (1. - eps) * G_z  # 相当于penalty分布 进行插值
    # grad = tf.gradients(discriminator(X_inter, y, reuse=True), [X_inter])[0]
    grad = tf.gradients(discriminator(X_inter, y), [X_inter])[0]
    grad_norm = tf.sqrt(
        tf.reduce_sum(grad ** 2, axis=1))  # tf.sqrt开方函数 tf.reduce_sum axis = 1代表行求行和，例如[[1,1,1],[1,1,1]]运行之后是[3,3]
    # grad_pen = 10 * tf.reduce_mean(tf.square(grad_norm - 1.))  # 斜率大于1才惩罚
    grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))  # 10为惩罚系数 斜率绝对值大于1就惩罚
    # 计算损失函数
    D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + grad_pen  # 最小化WGAN-GP
    # D_loss = tf.reduce_sum(tf.square(D_real_logits - 1) + tf.square(D_fake_logits)) / 2  # LSGAN
    G_loss = -tf.reduce_mean(D_fake_logits)  # 最小化WGAN-GP
    # G_loss = tf.reduce_sum(tf.square(D_fake_logits - 1)) / 2  # LSGAN
    print(type(G_loss))
    # 使用优化器更新参数
    # trainable variables for each network
    T_vars = tf.trainable_variables()  # tf.trainable_variables () 指的是需要训练的变量
    # D_vars = [var for var in T_vars if
    #           var.name.startswith('discriminator')]  # 返回变量名起始为discriminator的变量，实际找变量空间为discriminator内的变量
    # print(D_vars)<tf.Variable 'discriminator/fully_connected/weights:0' shape=(17, 64) dtype=float32_ref>
    # G_vars = [var for var in T_vars if var.name.startswith('generator')]
    # optimizer for each network
    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # tf.control_dependencies()该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行。这里是指保存工作完成之后在进行辖域中的操作
    # tf.get_collection(tf.GraphKeys.UPDATE_OPS)# 保存一些需要在训练操作之前完成的操作，这里指保存计算D_LOSS和G_LOSS之前需执行的操作
    D_optim = tf.train.RMSPropOptimizer(lr_D).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.RMSPropOptimizer(lr_g).minimize(G_loss, var_list=G_vars)
    # var_list: Optional list or tuple of `Variable` objects to update to minimize `loss`.

    # training-loop训练循环
    print('training start!')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    x_, y_ = load_data(select_number)  # 每类选50个样本，5类共250个
    # x_ = np.loadtxt('train_data50.txt')
    # y_ = np.loadtxt('train_labels50.txt')
    labels = one_hot(y_, M_size)  # 250个真实故障样本标签的one-hot表示
    for epoch in range(train_epoch):  # 20000次迭代
        epoch_start_time = time.time()

        # upadte Discriminator

        z_ = np.random.uniform(-1, 1, (G_size, Zn_size))  # 30*100 即30个生成故障样本，每个生成故障样本输入噪声为100维度
        for i in range(4):
            D_losses, _ = sess.run([D_loss, D_optim],
                                   {x: x_, y: labels, z: z_, gy: labels})  # CGAN的体现，加入标签元素

        # update generator
        z_ = np.random.uniform(-1, 1, (G_size, Zn_size))
        G_y, _ = G_labels(select_number, Zn_size)  # 返回one-hot表示的标签和对应的噪声
        G_losses, _ = sess.run([G_loss, G_optim],
                               {x: x_, y: labels, z: z_, gy: G_y})
        # 因为后两个返回的均是函数，返回命名同一个也可以，后期用不着返回值，只要执行了就好

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        G = show_result(epoch)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f'
          % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
    print("Training finish!... save training results")

    # results save folder
    root = 'data_results/'
    model = 'data_cGAN_'
    if not os.path.isdir(root):  # os.path.isdir()函数判断某一路径是否为目录
        os.mkdir(root)  # 没有此目录的话创建该目录
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    with open(root + model + 'train_hist.pkl', 'wb') as f:  # 可以略
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')