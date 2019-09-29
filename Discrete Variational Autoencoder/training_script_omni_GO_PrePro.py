# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:57:29 2018

@author: Yulai Cong
"""
import os, time, sys, datetime
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import datasets

LR = 1e-4
MBsize = 24
dim_var = [784, 200]
TestInterval = 5000
max_iters = 1000000

NonLinerNN = False
PreProcess = True
# dataset = "mnist"
dataset = "omni"

if dataset == "mnist":
    X_tr, X_va, X_te = datasets.load_mnist()
elif dataset == "omni":
    X_tr, X_va, X_te = datasets.load_omniglot()
else:
    assert False

num_train = X_tr.shape[0]
num_valid = X_va.shape[0]
num_test = X_te.shape[0]
train_mean = np.mean(X_tr, axis=0, keepdims=True)

tf.reset_default_graph()


def GOBernoulli(Prob):
    zsamp = tf.cast(tf.less_equal(tf.random_uniform(Prob.shape), Prob), tf.float32)
    zout = Prob + tf.stop_gradient(zsamp - Prob)
    return zout


def q_phi_z_gx_func(xin, out_dim, name, reuse=False):
    with tf.variable_scope('q_phi_' + name, reuse=reuse):
        if PreProcess:
            xin = (xin - train_mean)
        if NonLinerNN:
            h1 = tf.layers.dense(xin, out_dim, activation=tf.tanh)
            h2 = tf.layers.dense(h1, out_dim, activation=tf.tanh)
            zProbLogits = tf.layers.dense(h2, out_dim, activation=None)
        else:
            zProbLogits = tf.layers.dense(xin, out_dim, activation=None)
        zProb = tf.nn.sigmoid(zProbLogits)
        zsam = GOBernoulli(zProb)
        zProbLogits_S = tf.stop_gradient(zProbLogits)
        log_q_phi_z_gx = tf.reduce_sum(
            zsam * zProbLogits_S - tf.nn.softplus(zProbLogits_S),
            axis=1,
            keepdims=True
        )
        return zsam, zProb, log_q_phi_z_gx


def p_theta_z_func(zL, name, reuse=False):
    with tf.variable_scope('p_theta_' + name, reuse=reuse):
        zero_in = tf.zeros([zL.shape[0], 1])
        zProbLogits = tf.layers.dense(zero_in, zL.shape[1], activation=None)
        zProb = tf.nn.sigmoid(zProbLogits)
        log_p_theta_z = tf.reduce_sum(
            zL * zProbLogits - tf.nn.softplus(zProbLogits),
            axis=1,
            keepdims=True
        )
        return zProb, log_p_theta_z


def p_theta_x_gz_func(x, z, name, reuse=False):
    with tf.variable_scope('p_theta_' + name, reuse=reuse):
        if PreProcess:
            zin = tf.stop_gradient(2. * z - 1.)
        else:
            zin = tf.stop_gradient(z)
        if NonLinerNN:
            h1_bf_a = tf.layers.dense(zin, x.shape[1], activation=None, name='Layer1')
            h1 = tf.tanh(h1_bf_a)
            h2 = tf.layers.dense(h1, x.shape[1], activation=tf.tanh, name='Layer2')
            xProbLogits = tf.layers.dense(h2, x.shape[1], activation=None, name='Layer3')
        else:
            xProbLogits = tf.layers.dense(zin, x.shape[1], activation=None, name='Layer1')
        xProb = tf.nn.sigmoid(xProbLogits)
        log_p_theta_x_gz = tf.reduce_sum(
            x * xProbLogits - tf.nn.softplus(xProbLogits),
            axis=1,
            keepdims=True
        )

        a_z = tf.cast(tf.less_equal(z, 0.5), tf.float32) - tf.cast(tf.greater(z, 0.5), tf.float32)
        weights = tf.get_default_graph().get_tensor_by_name('p_theta_' + name + '/Layer1/kernel:0')
        if PreProcess:
            weights = weights * 2.
        if NonLinerNN:
            Xi_z_h1_bf_a = tf.expand_dims(h1_bf_a, axis=1) + tf.expand_dims(a_z, axis=2) * tf.expand_dims(weights,
                                                                                                          axis=0)
            Xi_z_h1 = tf.tanh(Xi_z_h1_bf_a)
            Xi_z_h2 = tf.layers.dense(Xi_z_h1, x.shape[1], activation=tf.tanh, name='Layer2', reuse=True)
            Xi_z_xProbLogits = tf.layers.dense(Xi_z_h2, x.shape[1], activation=None, name='Layer3', reuse=True)
        else:
            Xi_z_xProbLogits = tf.expand_dims(xProbLogits, axis=1) + tf.expand_dims(a_z, axis=2) * tf.expand_dims(
                weights, axis=0)

        Grad_z = a_z * (
                tf.reduce_sum(
                    tf.expand_dims(x, axis=1) * Xi_z_xProbLogits
                    - tf.nn.softplus(Xi_z_xProbLogits),
                    axis=2)
                - log_p_theta_x_gz
        )

        log_p_theta_x_gz_out = log_p_theta_x_gz + \
                               tf.reduce_sum(
                                   z * tf.stop_gradient(Grad_z),
                                   axis=1,
                                   keepdims=True
                               ) \
                               - tf.stop_gradient(tf.reduce_sum(z * Grad_z,
                                                                axis=1,
                                                                keepdims=True
                                                                )
                                                  )

    return xProb, log_p_theta_x_gz_out


# def p_theta_z_gz_func(zl, zlp1, name, reuse=False):
#     with tf.variable_scope('p_theta_' + name, reuse=reuse):
#         zlProbLogits = tf.layers.dense(tf.stop_gradient(zlp1), zl.shape[1], activation=None, name='Layer1')
#         zlProb = tf.nn.sigmoid(zlProbLogits)
#         log_p_theta_z_gz = tf.reduce_sum(
#             zl * zlProbLogits - tf.nn.softplus(zlProbLogits),
#             axis=1,
#             keepdims=True
#         )
#
#         a_zlp1 = tf.cast(tf.less_equal(zlp1, 0.5), tf.float32) - tf.cast(tf.greater(zlp1, 0.5), tf.float32)
#         weights = tf.get_default_graph().get_tensor_by_name('p_theta_' + name + '/Layer1/kernel:0')
#         Xi_zlp1_logits = tf.expand_dims(zlProbLogits, axis=1) + tf.expand_dims(a_zlp1 * weights, axis=0)
#         Grad_zlp1 = a_zlp1 * (
#                 tf.reduce_sum(
#                     tf.expand_dims(zl, axis=1) * Xi_zlp1_logits
#                     - tf.nn.softplus(Xi_zlp1_logits),
#                     axis=2)
#                 - log_p_theta_z_gz
#         )
#
#         log_p_theta_z_gz_out = log_p_theta_z_gz + \
#                                tf.reduce_sum(
#                                    zlp1 * tf.stop_gradient(Grad_zlp1),
#                                    axis=1,
#                                    keepdims=True
#                                ) \
#                                - tf.stop_gradient(tf.reduce_sum(zlp1 * Grad_zlp1,
#                                                                 axis=1,
#                                                                 keepdims=True
#                                                                 )
#                                                   )
#
#         return zlProb, log_p_theta_z_gz_out


# Training
xinput = tf.placeholder(tf.float32, shape=[MBsize, dim_var[0]])

z1, z1_Prob, log_q_phi_z_gx = q_phi_z_gx_func(xinput, dim_var[1], name='z1_gx', reuse=False)

xProb, log_p_theta_x_gz = p_theta_x_gz_func(xinput, z1, name='x_gz', reuse=False)

z1_Prob_pr, log_p_theta_z = p_theta_z_func(z1, name='z_pr', reuse=False)

loss = -tf.reduce_mean(log_p_theta_x_gz + log_p_theta_z - log_q_phi_z_gx)

# Validation
xinput_val = tf.placeholder(tf.float32, shape=[MBsize, dim_var[0]])

z1_val, _, log_q_phi_z_gx_val = q_phi_z_gx_func(xinput_val, dim_var[1], name='z1_gx', reuse=True)

_, log_p_theta_x_gz_val = p_theta_x_gz_func(xinput_val, z1_val, name='x_gz', reuse=True)

_, log_p_theta_z_val = p_theta_z_func(z1_val, name='z_pr', reuse=True)

loss_val = -tf.reduce_sum(log_p_theta_x_gz_val + log_p_theta_z_val - log_q_phi_z_gx_val)

# Test
xinput_test = tf.placeholder(tf.float32, shape=[MBsize, dim_var[0]])

z1_test, _, log_q_phi_z_gx_test = q_phi_z_gx_func(xinput_test, dim_var[1], name='z1_gx', reuse=True)

_, log_p_theta_x_gz_test = p_theta_x_gz_func(xinput_test, z1_test, name='x_gz', reuse=True)

_, log_p_theta_z_test = p_theta_z_func(z1_test, name='z_pr', reuse=True)

loss_test = -tf.reduce_sum(log_p_theta_x_gz_test + log_p_theta_z_test - log_q_phi_z_gx_test)

# Reconstruction
# Sampling fake data

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR)

training = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Results saving
result_dir = './Results_MNIST_SBN'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if NonLinerNN:
    if PreProcess:
        shutil.copyfile(sys.argv[0], result_dir + '/training_script_' + dataset + '_GO_NonLinear_PrePro.py')
        pathsave = result_dir + '/TF_SBN_' + dataset + '_GO_NonLinear_PrePro_L[%d_%d]_LR[%.2e].mat' % (dim_var[0], dim_var[1], LR)
    else:
        shutil.copyfile(sys.argv[0], result_dir + '/training_script_' + dataset + '_GO_NonLinear.py')
        pathsave = result_dir + '/TF_SBN_' + dataset + '_GO_NonLinear_L[%d_%d]_LR[%.2e].mat' % (dim_var[0], dim_var[1], LR)
else:
    if PreProcess:
        shutil.copyfile(sys.argv[0], result_dir + '/training_script_' + dataset + '_GO_PrePro.py')
        pathsave = result_dir + '/TF_SBN_' + dataset + '_GO_PrePro_L[%d_%d]_LR[%.2e].mat' % (dim_var[0], dim_var[1], LR)
    else:
        shutil.copyfile(sys.argv[0], result_dir + '/training_script_' + dataset + '_GO.py')
        pathsave = result_dir + '/TF_SBN_' + dataset + '_GO_L[%d_%d]_LR[%.2e].mat' % (dim_var[0], dim_var[1], LR)

tr_loss_mb_set = []
tr_timerun_mb_set = []
tr_iter_mb_set = []

tr_loss_set = []
tr_timerun_set = []
tr_iter_set = []

val_loss_set = []
val_timerun_set = []
val_iter_set = []

te_loss_set = []
te_timerun_set = []
te_iter_set = []

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(init)

    for epoch in range(10000000):

        iters_per_epoch = X_tr.shape[0] // MBsize

        for i in range(iters_per_epoch):

            step = epoch * iters_per_epoch + i

            if step > max_iters:
                break

            if step == 0:
                time_start = time.clock()

            # batch_x, _ = mnist.train.next_batch(MBsize)
            # batch_x[batch_x > 0.5] = 1.
            # batch_x[batch_x <= 0.5] = 0.

            batch_x = X_tr[i * MBsize: (i + 1) * MBsize]

            feed_dict = {xinput: batch_x}

            _, loss1 = sess.run([training, loss], feed_dict=feed_dict)

            time_run = time.clock() - time_start

            tr_loss_mb_set.append(loss1)
            tr_timerun_mb_set.append(time_run)
            tr_iter_mb_set.append(step + 1)

            if (step + 1) % 100 == 0:
                print('Step: [{:6d}], Loss: [{:10.4f}], time_run: [{:10.4f}]'.format(step + 1, loss1, time_run))

            # Testing
            Train_num_mbs = num_train // MBsize
            Valid_num_mbs = num_valid // MBsize
            Test_num_mbs = num_test // MBsize

            if (step + 1) % TestInterval == 0:

                # Training
                loss_train1 = 0
                for step_train in range(Train_num_mbs):
                    x_train = X_tr[step_train * MBsize: (step_train + 1) * MBsize]

                    feed_dict_train = {xinput: x_train}
                    loss_train_mb1 = sess.run(loss, feed_dict=feed_dict_train)
                    loss_train1 += loss_train_mb1 * MBsize

                loss_train1 = loss_train1 / (Train_num_mbs * MBsize)

                tr_loss_set.append(loss_train1)
                tr_timerun_set.append(time_run)
                tr_iter_set.append(step + 1)

                # Validation
                loss_val1 = 0
                for step_val in range(Valid_num_mbs):
                    # x_valid, _ = mnist.validation.next_batch(100)
                    # x_valid[x_valid > 0.5] = 1.
                    # x_valid[x_valid <= 0.5] = 0.

                    x_valid = X_va[step_val * MBsize: (step_val + 1) * MBsize]

                    feed_dict_val = {xinput_val: x_valid}
                    loss_val_mb1 = sess.run(loss_val, feed_dict=feed_dict_val)
                    loss_val1 += loss_val_mb1

                loss_val1 = loss_val1 / (Valid_num_mbs * MBsize)

                val_loss_set.append(loss_val1)
                val_timerun_set.append(time_run)
                val_iter_set.append(step + 1)

                # Test
                loss_test1 = 0
                for step_test in range(Test_num_mbs):
                    # x_test, _ = mnist.test.next_batch(100)
                    # x_test[x_test > 0.5] = 1.
                    # x_test[x_test <= 0.5] = 0.

                    x_test = X_te[step_test * MBsize: (step_test + 1) * MBsize]

                    feed_dict_test = {xinput_test: x_test}
                    loss_test_mb1 = sess.run(loss_test, feed_dict=feed_dict_test)
                    loss_test1 += loss_test_mb1

                loss_test1 = loss_test1 / (Test_num_mbs * MBsize)

                te_loss_set.append(loss_test1)
                te_timerun_set.append(time_run)
                te_iter_set.append(step + 1)

                print(
                    '============TestInterval: [{:6d}], Loss_train: [{:10.4f}], Loss_val: [{:10.4f}], Loss_test: [{:10.4f}]'.format(
                        TestInterval, loss_train1, loss_val1, loss_test1))

            # Saving
            if (step + 1) % TestInterval == 0:
                sio.savemat(pathsave, {'tr_loss_mb_set': tr_loss_mb_set,
                                       'tr_timerun_mb_set': tr_timerun_mb_set,
                                       'tr_iter_mb_set': tr_iter_mb_set,

                                       'tr_loss_set': tr_loss_set,
                                       'tr_timerun_set': tr_timerun_set,
                                       'tr_iter_set': tr_iter_set,

                                       'val_loss_set': val_loss_set,
                                       'val_timerun_set': val_timerun_set,
                                       'val_iter_set': val_iter_set,

                                       'te_loss_set': te_loss_set,
                                       'te_timerun_set': te_timerun_set,
                                       'te_iter_set': te_iter_set,
                                       })

        if step > max_iters:
            print("Training Completed - max_iters[%d]" % max_iters)
            break