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
from option import Options

tf.reset_default_graph()
# training parameters
op = Options()
level_num = op.level_num
select_number = op.select_number
M_size = select_number * 7
N_size = 65
LabN_size = 7
G_size = select_number * 7
Zn_size = 100
lr_g = 0.0001
lr_D = 0.0001
train_epoch = 20000
train_hist = dict()  # train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
# variables : input
x = tf.placeholder(tf.float32, shape=(None, N_size))
y = tf.placeholder(tf.float32, shape=(None, LabN_size))
z = tf.placeholder(tf.float32, shape=(None, Zn_size))
gy = tf.placeholder(tf.float32, shape=(None, LabN_size))


def load_data(select_num, name1="train_data"+str(select_number)+"_level"+str(level_num)+"_normalization.txt", name2="train_labels"+str(select_number)+".txt"):  # select_num = 50 #选择每类选取的故障样本数
    size = select_num * 7   # 共7类
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
    train_labels = train_data[:, 0].reshape(size, 1)  # 标签
    train_data = np.delete(train_data, [0], axis=1)  # 删除第一列
    np.savetxt("train_data" + str(select_number) + "_level"+str(level_num)+"_original.txt",
               train_data)
    min = np.min(train_data, axis=0)
    max = np.max(train_data, axis=0)
    train_data = MaxMinNormalization(train_data, min, max)
    np.savetxt(name1, train_data)
    np.savetxt(name2, train_labels)
    return train_data, train_labels


def MaxMinNormalization(x, min, max):
    x = (x - min) / (max - min + 0.0000001)
    return x

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# G(z)
def generator(x, y, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)
        z = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 256, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 128, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 65, activation_fn=tf.nn.relu)
        return z


# D(x)
def discriminator(x, y, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)
        x = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 1, activation_fn=None)
        return x


def one_hot(y, size):
    label = []
    for i in range(size):
        a = int(y[i]) - 1
        temp = [0, 0, 0, 0, 0, 0, 0]
        temp[a] = 1
        label.extend(temp)
    label = np.array(label).reshape(size, 7)
    return label


def G_labels(select_num, size, set_name=False):
    t = 6
    temp_z_ = np.random.uniform(-1, 1, (select_num, size))
    z_ = temp_z_
    fixed_y_ = np.ones((select_num, 1))
    j = 1
    for i in range(t):
        temp = np.ones((select_num, 1)) + j
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)
    y = one_hot(fixed_y_, select_num * 7)
    if set_name:
        name = "labels1500" + ".txt"
        np.savetxt(name, fixed_y_)
    return y, z_


def show_result(epoch_num):
    with tf.variable_scope('show_result'):
        if epoch_num == 19999:  # 19999
            G_y, fixed_z_ = G_labels(1500, 100, True)
            G = sess.run(G_z, {z: fixed_z_, gy: G_y})
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
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


G_z = generator(z, gy)
D_real_logits = discriminator(x, y)
D_fake_logits = discriminator(G_z, y, reuse=True)
eps = tf.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)
X_inter = eps * x + (1. - eps) * G_z
grad = tf.gradients(discriminator(X_inter, y, reuse=True), [X_inter])[0]
grad_norm = tf.sqrt(
    tf.reduce_sum(grad ** 2, axis=1))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))
D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + grad_pen
G_loss = -tf.reduce_mean(D_fake_logits)
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if
          var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]
# optimizer for each network
D_optim = tf.train.RMSPropOptimizer(lr_D).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.RMSPropOptimizer(lr_g).minimize(G_loss, var_list=G_vars)
print('training start!')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start_time = time.time()
x_, y_ = load_data(select_number)
labels = one_hot(y_, M_size)
for epoch in range(train_epoch):
    epoch_start_time = time.time()
    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))
    for i in range(4):
        D_losses, _ = sess.run([D_loss, D_optim],
                               {x: x_, y: labels, z: z_, gy: labels})
    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))
    G_y, _ = G_labels(select_number, Zn_size)
    G_losses, _ = sess.run([G_loss, G_optim],
                           {x: x_, y: labels, z: z_, gy: G_y})
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
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

