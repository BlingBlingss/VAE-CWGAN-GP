# 使用CWGAN生成故障样本
# 提取出归一化返回均值，最大值减最小值的值

# -*- coding: utf-8 -*-
"""
Created on  June  5 10:54:44 2019

@author: Jianye Su
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
level_num = 1  # 等级
select_number = 10  # 选择每类选取的故障样本数
M_size = select_number * 7  # 7类真实故障样本共M_size个
N_size = 65  # 选取的特征数

LabN_size = 7  # onehot表示标签向量

G_size = select_number * 7  # 生成故障样本数G_size个
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
x = tf.placeholder(tf.float32, shape=(None, N_size))  # 65个特征
y = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 7个one-hot表示的标签
z = tf.placeholder(tf.float32, shape=(None, Zn_size))  # 100
gy = tf.placeholder(tf.float32, shape=(None, LabN_size))  # 7


# 加载真实故障数据
def load_data(select_num, name1="train_data"+str(select_number)+"_level"+str(level_num)+"_normalization.txt", name2="train_labels"+str(select_number)+".txt"):  # select_num = 50 #选择每类选取的故障样本数

    size = select_num * 7   # 共7类
    total_num = 36337
    step_size = 5191
    TR_sample_temp = sio.loadmat("Level"+str(level_num)+".mat")
    sample = TR_sample_temp['num']  # 每类5191个，故数据总量为36337*66，第一列为类标

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
    # np.savetxt("train_data"+str(select_number)+"_original.txt", train_data)  # 存储归一化之前的原始的真实样本，用于计算其中的最小值与最大值从而对训练集进行归一化
    np.savetxt("train_data" + str(select_number) + "_level"+str(level_num)+"_original.txt",
               train_data)  # 存储归一化之前的原始的真实样本，用于计算其中的最小值与最大值从而对训练集进行归一化
    # Normalized processing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_data1 = min_max_scaler.fit_transform(train_data)
    # print("hello", train_data1)
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)
    # print("min:", min)
    # print("max:", max)
    train_data = MaxMinNormalization(train_data, min, max)
    np.savetxt(name1, train_data)  # 存储归一化之后的原始的真实样本
    np.savetxt(name2, train_labels)
    return train_data, train_labels


def MaxMinNormalization(x, min, max):
    x = (x - min) / (max - min + 0.0000001)
    return x

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)  # 0.6
    f2 = 0.5 * (1 - leak)  # 0.4
    return f1 * X + f2 * tf.abs(X)  # 即f(x) = 0.6x + 0.4|x|


# G(z)
def generator(x, y, reuse=False):  # 注意这里把标签信息y作为输入,从而成为CGAN

    with tf.variable_scope('generator', reuse=reuse):
        # # 定义变量空间的名称generator，Variables created here will be named "generator/var1", "generator/var2".
        cat1 = tf.concat([x, y], 1)  # 比如两个shape为[2,3]的矩阵拼接，可以通过axis=0变成[4,3]，或者通过axis=1变成[2,6]。
        # CGAN加入类别信息
        # z = slim.fully_connected(cat1, 64, activation_fn=tf.nn.relu)
        z = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)  # 参数分别为网络输入、输出的神经元数量，激活函数
        z = slim.fully_connected(z, 256, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 128, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 65, activation_fn=tf.nn.relu)
        return z


# D(x)
def discriminator(x, y, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)
        # tf.nn.relu()函数是将大于0的数保持不变，小于0的数置为0
        x = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 1, activation_fn=None)  # WGAN
        # x = slim.fully_connected(x, 1, activation_fn=tf.nn.leaky_relu) # lsgan
        # x = dense(cat1, 64, activation=tf.nn.relu, sn=True)
        # x = dense(x, 128, activation=tf.nn.relu, sn=True)
        # x = dense(x, 64, activation=tf.nn.relu, sn=True)
        # x = dense(x, 32, activation=tf.nn.relu, sn=True)
        # x = dense(x, 1, activation=None)
        # 不做任何非线性处理，activation=none，因为计算Wasserstein Distance不需要sigmoid函数

        return x


def one_hot(y, size):  # y为实际标签，size为样本个数
    label = []
    for i in range(size):  # size为样本个数
        a = int(y[i]) - 1
        temp = [0, 0, 0, 0, 0, 0, 0]  # 共7类故障
        temp[a] = 1
        label.extend(temp)  # 在列表末尾一次性追加另一个序列中的多个值
    label = np.array(label).reshape(size, 7)  # 最后重塑
    return label


# 返回one-hot表示的标签和对应的噪声
def G_labels(select_num, size, set_name=False):  # size为噪声维度

    t = 6
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
    y = one_hot(fixed_y_, select_num * 7)  # 7类
    if set_name:
        name = "labels1500" + ".txt"
        np.savetxt(name, fixed_y_)
    return y, z_  # 返回one-hot表示的标签和对应的噪声


def show_result(epoch_num):  # 网络训练好后生成最终的故障样本
    with tf.variable_scope('show_result'):
        if epoch_num == 19999:  # 19999
            G_y, fixed_z_ = G_labels(1500, 100, True)  # 返回顺序排列的生成标签和对应的噪声共10500个
            G = sess.run(G_z, {z: fixed_z_, gy: G_y})  # 返回含有65个特征的生成样本
            G_sample = G
            name = "CWGAN_data" + str(select_number) + "_level" + str(level_num) + ".txt"
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


# if __name__ == "__main__":
# 1、定义计算图所有的计算
# networks : generator
G_z = generator(z, gy)  # 返回生成的含有65个特征的生成样本

# networks : discriminator
D_real_logits = discriminator(x, y)  # 原数据评分,使用默认的reuse=False先创建变量
D_fake_logits = discriminator(G_z, y, reuse=True)  # 生成数据评分，reuse = True复用变量

# 计算损失函数
# Wgan trick 加入惩罚项
eps = tf.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)  # 返回一个矩阵，产生于minval和maxval之间，产生的值是均匀分布的。G_size为30
X_inter = eps * x + (1. - eps) * G_z  # 相当于penalty分布 进行插值
grad = tf.gradients(discriminator(X_inter, y, reuse=True), [X_inter])[0]
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
D_vars = [var for var in T_vars if
          var.name.startswith('discriminator')]  # 返回变量名起始为discriminator的变量，实际找变量空间为discriminator内的变量
# print(D_vars)<tf.Variable 'discriminator/fully_connected/weights:0' shape=(17, 64) dtype=float32_ref>
G_vars = [var for var in T_vars if var.name.startswith('generator')]
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
x_, y_ = load_data(select_number)  # 每类选50个样本，7类共350个
#load_data(select_number, "test_data50_new" + ".txt", "test_labels50_new" + ".txt", min_train_data, max_train_data)
# x_ = np.loadtxt('train_data50.txt')
# y_ = np.loadtxt('train_labels50.txt')
labels = one_hot(y_, M_size)  # 350个真实故障样本标签的one-hot表示
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

