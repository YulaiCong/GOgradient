import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio

np.set_printoptions(formatter={'float': '{: .1e}'.format})

"""    Clip Gradients
Created on Wed Apr 14 21:01:53 2018
@author: Yulai Cong
"""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./Data/MNIST/", one_hot=True)

tf.reset_default_graph()

#   sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Training Params
num_steps = 20000
batch_size = 200
rho_mb = mnist.train.num_examples / batch_size

# Network Params
MethodName = 'GO'
#MethodName = 'GRep'
#MethodName = 'RSVI'

Layers = 2

K = [784., 128., 64., 32.]
h_dim = [0, 1024, 256, 128]
alpha_z = 0.1

LR_q_z1 = 5e-5  # 5e-4
LR_q_W1 = 1.5  # 2e-1
LR_q_z2 = 1e-5  # 5e-4
LR_q_W2 = 0.5  # 2e-1
LR_q_W3 = 1.5  # 2e-1

min_z = 1e-5
min_W = 1e-5

min_z_alpha = 1e-3
min_z_beta = 1e-3
min_W_alpha = 1e-3
min_W_beta = 1e-3  # min_mean = 1e-4

min_z_alpha_rate = float(np.log(np.exp(min_z_alpha) - 1.))
min_z_beta_rate = float(np.log(np.exp(min_z_beta) - 1.))
min_W_alpha_rate = float(np.log(np.exp(min_W_alpha) - 1.))
min_W_beta_rate = float(np.log(np.exp(min_W_beta) - 1.))

# Compared method parameters
B = 5
Bf = 5.


def reject_h_boosted(p, alpha):
    # compute eps
    alpha_jian = alpha + B
    sqrtAlpha = tf.sqrt(9. * alpha_jian - 3.)
    t = alpha_jian - 1. / 3.
    powZA = tf.pow((p / t), 1. / 3.)
    eps = tf.stop_gradient( sqrtAlpha * (powZA - 1.))
    
    b = (alpha_jian) - 1. / 3.
    c = 1. / tf.sqrt(9. * b)
    v = 1. + eps * c
    v = tf.sign(v)*tf.maximum(tf.abs(v),1e-7)
    z_jian = b * tf.pow(v, 3.)
    z_jian = tf.maximum(z_jian,1e-7)
    # compute z_bo
    ni = alpha.shape[0]
    ki = alpha.shape[1]
    alpha = alpha[:, :, tf.newaxis]
    tmp = tf.range(Bf)
    tmp = tmp[tf.newaxis, tf.newaxis, :]
    alpha_vec = tf.tile(alpha, [1, 1, B]) + tf.tile(tmp, [ni, ki, 1])
    u = tf.maximum(tf.random_uniform([int(ni), int(ki), B]), 1e-7)
    u_pow = tf.pow(u, 1. / alpha_vec)
    
    z_bo = tf.keras.backend.prod(u_pow, axis=2) * z_jian
    # g_corr
    log_q = - tf.lgamma(alpha_jian) + (alpha_jian - 1.) * tf.log(z_jian) - z_jian
    log_PzPeps = tf.log(3. * b) + 2 * tf.log(v) - 0.5 * tf.log(9. * b)
    f_corr = tf.reduce_sum(log_q + log_PzPeps)
    return z_bo, f_corr


# Recognition Model - q_z_x
def q_z_x(name, x, K, reuse=False):
    with tf.variable_scope('q_z_x' + name, reuse=reuse):
        # h1 = tf.nn.relu(tf.layers.dense(x, units=h_dim[1]))
        h1 = x

        z_Ralpha1 = tf.layers.dense(h1, units=K, kernel_initializer=tf.random_normal_initializer(0, 0.01))
        z_Ralpha1 = max_m_grad(min_z_alpha_rate, z_Ralpha1)
        # z_Ralpha1 = tf.maximum(min_z_alpha_rate, z_Ralpha1)
        z_alpha1 = tf.nn.softplus(z_Ralpha1)

        z_Rbeta1 = tf.layers.dense(h1, units=K, kernel_initializer=tf.random_normal_initializer(0, 0.01))
        z_Rbeta1 = max_m_grad(min_z_beta_rate, z_Rbeta1)
        # z_Rbeta1 = tf.maximum(min_z_beta_rate, z_Rbeta1)
        z_beta1 = tf.nn.softplus(z_Rbeta1)
        # z_beta1 = min_m_grad(z_alpha1 / min_mean, z_beta1)

        if MethodName == 'GO':
            z_hat1s = tf.random_gamma([1], tf.stop_gradient(z_alpha1), 1.)
            z_hat1s = tf.maximum(min_z, tf.squeeze(z_hat1s, 0))
            Grad_z_alpha1 = GO_Gamma_v2(tf.stop_gradient(z_hat1s), tf.stop_gradient(z_alpha1))
            z_hat1 = z_alpha1 * tf.stop_gradient(Grad_z_alpha1) - \
                     tf.stop_gradient(z_alpha1 * Grad_z_alpha1) + \
                     tf.stop_gradient(z_hat1s)
            z1_Fcorr = tf.zeros([1])

        if MethodName == 'GRep':
            posi0 = tf.polygamma(tf.constant(0,dtype=tf.float32),z_alpha1)
            posi1 = tf.polygamma(tf.constant(1,dtype=tf.float32),z_alpha1)
            z_hat1s = tf.random_gamma([1], tf.stop_gradient(z_alpha1), 1.)
            z_hat1s = tf.maximum(min_z, tf.squeeze(z_hat1s, 0))
            epsilo = tf.stop_gradient( (tf.log(z_hat1s)-posi0)/tf.maximum((tf.pow(posi1,0.5)),1e-5) )
            log_z_hat1 = epsilo*tf.pow(posi1,0.5)+posi0
            z_hat1 = tf.exp( log_z_hat1 )
            z1_Fcorr = tf.reduce_sum(
                    - tf.lgamma(z_alpha1) + (z_alpha1-1.)*log_z_hat1 - z_hat1
                    + log_z_hat1 + 0.5 * tf.log( posi1 )
                    ) 

        if MethodName == 'RSVI':
            lambda_z1 = tf.squeeze(tf.random_gamma([1], z_alpha1 + Bf, 1.), 0)
            lambda_z1 = tf.stop_gradient(tf.maximum(min_z, lambda_z1))
            z_hat1, z1_Fcorr = reject_h_boosted(lambda_z1, z_alpha1)

        z1 = z_hat1 / z_beta1
        # z1 = tf.maximum(min_z, z1)
        z1 = max_m_grad(min_z, z1)

        return z1, z_alpha1, z_beta1, z1_Fcorr


def max_m_grad(epsi, x):
    y = tf.maximum(epsi, x)
    yout = x - tf.stop_gradient(x) + tf.stop_gradient(y)
    return yout


def min_m_grad(epsi, x):
    y = tf.minimum(epsi, x)
    yout = x - tf.stop_gradient(x) + tf.stop_gradient(y)
    return yout


def map_Dir_64(Phi, Eps):
    # Phi: V * K
    Eps = tf.cast(Eps, tf.float64)
    Phi = tf.cast(Phi, tf.float64)
    PhiT = tf.transpose(Phi)
    Kphi = PhiT.shape[0]
    Vphi = PhiT.shape[1]

    PhiTsort, _ = tf.nn.top_k(PhiT, Vphi)
    CumPhiT = tf.cumsum(PhiTsort, 1)

    i_v = tf.range(1, tf.cast(Vphi, tf.float64), dtype=tf.float64)
    tmp = CumPhiT[:, :Vphi - 1] - tf.expand_dims(i_v, 0) * PhiTsort[:, 1:]
    tmp1 = tf.to_float(tmp >= (1. - tf.cast(Vphi, tf.float64) * Eps))
    B, I = tf.nn.top_k(tmp1)
    B = tf.cast(B, tf.float64)
    I = tf.cast(I, tf.float64)
    I = I + (1. - B) * (tf.cast(Vphi, tf.float64) - 1.)

    indx0 = tf.range(Kphi, dtype=tf.int64)
    indx = tf.concat([indx0[:, tf.newaxis], tf.cast(I, tf.int64)], axis=1)

    delta = (1. - Eps * (tf.cast(Vphi, tf.float64) - I - 1.) - tf.expand_dims(tf.gather_nd(CumPhiT, indx), 1)) / (
            I + 1.)
    Phihat = tf.maximum(Eps, Phi + tf.transpose(delta))

    Phiout = Phi - tf.stop_gradient(Phi) + tf.stop_gradient(Phihat)

    Phiout = tf.cast(Phiout, tf.float32)
    return Phiout


# GO Gamma Gradients
def GO_Gamma_v2(x, alpha):
    # x sim Gamma(alpha, 1)
    x = tf.cast(x, tf.float64)
    alpha = tf.cast(alpha, tf.float64)

    logx = tf.log(x)
    ex_gamma_xa = tf.exp(x + tf.lgamma(alpha) + (1. - alpha) * logx)
    psi_m_log = tf.digamma(alpha + 1.) - logx
    igamma_up_reg = tf.igammac(alpha, x)

    # Part 1
    indx1 = tf.where(x <= 1e-2)
    x_indx1 = tf.gather_nd(x, indx1)
    alpha_indx1 = tf.gather_nd(alpha, indx1)
    GO_Gamma_alpha_value1 = tf.exp(x_indx1) * x_indx1 / alpha_indx1 * (
            tf.gather_nd(psi_m_log, indx1) +
            x_indx1 / tf.pow(alpha_indx1 + 1., 2) -
            tf.pow(x_indx1 / (alpha_indx1 + 2.), 2) +
            0.5 * tf.pow(x_indx1, 3) / tf.pow(alpha_indx1 + 3., 2)
    )

    # Part 2
    N_alpha = tf.round(tf.exp(
        - 0.488484605941243044124888683654717169702053070068359375 * tf.log(alpha)
        + 1.6948389987594634220613443176262080669403076171875
    ))
    indx2 = tf.where(tf.logical_and(
        tf.logical_and(x > 1e-2, alpha <= 3.),
        (x <= (alpha + N_alpha * tf.sqrt(alpha)))
    ))
    KK = 15
    kk = tf.cast(tf.range(1, KK + 1), tf.float64)
    x_indx2 = tf.gather_nd(x, indx2)
    alpha_indx2 = tf.gather_nd(alpha, indx2)
    GO_Gamma_alpha_value2 = tf.gather_nd(ex_gamma_xa, indx2) * (
            -tf.gather_nd(psi_m_log, indx2) * tf.gather_nd(igamma_up_reg, indx2) +
            (
                    tf.digamma(alpha_indx2 + KK + 1.) - tf.gather_nd(logx, indx2) -
                    tf.reduce_sum(
                        tf.igammac(tf.expand_dims(alpha_indx2, 1) + tf.expand_dims(kk, 0), tf.expand_dims(x_indx2, 1)) /
                        (tf.expand_dims(alpha_indx2, 1) + tf.expand_dims(kk, 0))
                        , 1)
            )
    )

    # Part 2_1
    indx2_1 = tf.where(tf.logical_and(
        tf.logical_and(x > 1e-2, alpha <= 3.),
        (x > (alpha + N_alpha * tf.sqrt(alpha)))
    ))
    KK = 15
    kk = tf.cast(tf.range(1, KK + 1), tf.float64)
    x_indx2_1 = tf.gather_nd(x, indx2_1)
    alpha_indx2_1 = tf.gather_nd(alpha, indx2_1)
    GO_Gamma_alpha_value2_1 = tf.gather_nd(ex_gamma_xa, indx2_1) * (
            -tf.gather_nd(psi_m_log, indx2_1) * tf.gather_nd(igamma_up_reg, indx2_1) +
            (
                    tf.digamma(alpha_indx2_1 + KK + 1.) - tf.gather_nd(logx, indx2_1) -
                    tf.reduce_sum(
                        tf.igammac(tf.expand_dims(alpha_indx2_1, 1) + tf.expand_dims(kk, 0),
                                   tf.expand_dims(x_indx2_1, 1)) /
                        (tf.expand_dims(alpha_indx2_1, 1) + tf.expand_dims(kk, 0))
                        , 1)
            )
    )
    GO_Gamma_alpha_value2_1 = tf.maximum(
        GO_Gamma_alpha_value2_1,
        1. / alpha_indx2_1 - tf.gather_nd(ex_gamma_xa, indx2_1) *
        tf.gather_nd(psi_m_log, indx2_1) * tf.gather_nd(igamma_up_reg, indx2_1)
    )

    # Part 3
    indx3 = tf.where(
        tf.logical_and(
            tf.logical_and(x > 1e-2, alpha > 3.),
            alpha <= 500.
        )
    )
    KK = 10
    kk = tf.cast(tf.range(1, KK + 1), tf.float64)
    x_indx3 = tf.gather_nd(x, indx3)
    alpha_indx3 = tf.gather_nd(alpha, indx3)

    x_l = alpha_indx3 - tf.log(alpha_indx3) * tf.sqrt(alpha_indx3)

    logx_l = tf.log(x_l)
    ex_gamma_xa_l = tf.exp(x_l + tf.lgamma(alpha_indx3) + (1. - alpha_indx3) * logx_l)
    psi_m_log_l = tf.digamma(alpha_indx3 + 1.) - logx_l
    igamma_low_reg_l = tf.igamma(alpha_indx3, x_l)
    # igamma_up_reg_l = tf.igammac(alpha_indx3, x_l)
    # f_l = ex_gamma_xa_l * (
    #         -psi_m_log_l * igamma_up_reg_l +
    #         (tf.digamma(alpha_indx3 + KK + 1.) - logx_l -
    #         tf.reduce_sum(
    #             tf.igammac(tf.expand_dims(alpha_indx3, 1) + tf.expand_dims(kk, 0), tf.expand_dims(x_l, 1)) /
    #             (tf.expand_dims(alpha_indx3, 1) + tf.expand_dims(kk, 0))
    #             , 1))
    # )
    f_l = ex_gamma_xa_l * (
            psi_m_log_l * igamma_low_reg_l +
            tf.reduce_sum(
                tf.igamma(tf.expand_dims(alpha_indx3, 1) + tf.expand_dims(kk, 0), tf.expand_dims(x_l, 1)) /
                (tf.expand_dims(alpha_indx3, 1) + tf.expand_dims(kk, 0))
                , 1)
    )

    g_l = (1. + (1. - alpha_indx3) / x_l) * f_l + (
            -ex_gamma_xa_l / x_l * igamma_low_reg_l + (psi_m_log_l +
                                                       tf.reduce_sum(
                                                           tf.exp(
                                                               tf.expand_dims(kk, 0) * tf.log(tf.expand_dims(x_l, 1)) +
                                                               tf.lgamma(tf.expand_dims(alpha_indx3, 1)) -
                                                               tf.lgamma(
                                                                   tf.expand_dims(alpha_indx3, 1) + tf.expand_dims(kk,
                                                                                                                   0) + 1.)
                                                           )
                                                           , 1))
    )

    x_m = alpha_indx3

    f_m = 1. + 0.167303227226226980395296095593948848545551300048828125 / \
          (
                  tf.pow(x_m, 1.0008649793164192676186985409003682434558868408203125) -
                  0.07516433982238841793321881823430885560810565948486328125
          )

    x_r = 2. * alpha_indx3 - x_l

    f_r = 1. / alpha_indx3 - tf.exp(x_r + tf.lgamma(alpha_indx3) + (1. - alpha_indx3) * tf.log(x_r)) * (
            (tf.digamma(alpha_indx3 + 1.) - tf.log(x_r)) * tf.igammac(alpha_indx3, x_r)
    )
    lambda_r = tf.exp(
        959.627335718427275423891842365264892578125 / (
                tf.pow(alpha_indx3, 1.324768828487964622553363369661383330821990966796875) +
                142.427456986662718918523751199245452880859375
        )
        - 13.01439996187340142341781756840646266937255859375
    )

    x_mat_i = tf.concat([tf.expand_dims(x_l, 1), tf.expand_dims(x_m, 1), tf.expand_dims(x_r, 1)], 1)
    x_mat_bar_i = x_mat_i - tf.expand_dims(alpha_indx3, 1)
    x_mat_hat_i = tf.sqrt(x_mat_i) - tf.sqrt(tf.expand_dims(alpha_indx3, 1))
    f_mat_i = tf.concat([tf.expand_dims(f_l, 1), tf.expand_dims(f_m, 1), tf.expand_dims(f_r, 1)], 1)
    lambda_mat_i = tf.concat([tf.expand_dims(tf.ones_like(alpha_indx3), 1),
                              tf.expand_dims(tf.ones_like(alpha_indx3), 1),
                              tf.expand_dims(lambda_r, 1)
                              ], 1)

    x_mat_j = tf.expand_dims(x_l, 1)
    g_mat_j = tf.expand_dims(g_l, 1)
    lambda_mat_j = tf.expand_dims(tf.ones_like(alpha_indx3), 1)

    A = tf.reduce_sum(lambda_mat_i * tf.pow(x_mat_bar_i, 2), 1) + tf.reduce_sum(lambda_mat_j, 1)
    B = tf.reduce_sum(lambda_mat_i * x_mat_bar_i * x_mat_hat_i, 1) + \
        tf.reduce_sum(lambda_mat_j / 2. / tf.sqrt(x_mat_j), 1)
    C = tf.reduce_sum(lambda_mat_i * x_mat_bar_i, 1)
    D = tf.reduce_sum(lambda_mat_i * tf.pow(x_mat_hat_i, 2), 1) + tf.reduce_sum(lambda_mat_j / 4. / x_mat_j, 1)
    E = tf.reduce_sum(lambda_mat_i * x_mat_hat_i, 1)
    F = tf.reduce_sum(lambda_mat_i, 1)
    G = tf.reduce_sum(lambda_mat_i * x_mat_bar_i * f_mat_i, 1) + tf.reduce_sum(lambda_mat_j * g_mat_j, 1)
    H = tf.reduce_sum(lambda_mat_i * x_mat_hat_i * f_mat_i, 1) + \
        tf.reduce_sum(lambda_mat_j / 2. / tf.sqrt(x_mat_j) * g_mat_j, 1)
    I = tf.reduce_sum(lambda_mat_i * f_mat_i, 1)

    Z = F * tf.pow(B, 2) - 2. * B * C * E + D * tf.pow(C, 2) + A * tf.pow(E, 2) - A * D * F

    a_cor = 1. / Z * (G * (tf.pow(E, 2) - D * F) + H * (B * F - C * E) - I * (B * E - C * D))
    b_cor = 1. / Z * (G * (B * F - C * E) + H * (tf.pow(C, 2) - A * F) - I * (B * C - A * E))
    c_cor = 1. / Z * (-G * (B * E - C * D) + I * (tf.pow(B, 2) - A * D) - H * (B * C - A * E))

    GO_Gamma_alpha_value3 = a_cor * (x_indx3 - alpha_indx3) + b_cor * (tf.sqrt(x_indx3) - tf.sqrt(alpha_indx3)) + c_cor
    GO_Gamma_alpha_value3 = tf.maximum(
        GO_Gamma_alpha_value3,
        1. / alpha_indx3 - tf.gather_nd(ex_gamma_xa, indx3) *
        tf.gather_nd(psi_m_log, indx3) * tf.gather_nd(igamma_up_reg, indx3)
    )

    # Part 4
    # indx4 = tf.where(
    #     tf.logical_and(
    #         tf.logical_and(x > 1e-2, alpha > 500.),
    #         (x <= (alpha + 2. * tf.log(alpha) * tf.sqrt(alpha)))
    #     )
    # )
    indx4 = tf.where(
        tf.logical_and(x > 1e-2, alpha > 500.)
    )

    x_indx4 = tf.gather_nd(x, indx4)
    alpha_indx4 = tf.gather_nd(alpha, indx4)

    f_m_large = 1. + 0.167303227226226980395296095593948848545551300048828125 / \
                (
                        tf.pow(alpha_indx4, 1.0008649793164192676186985409003682434558868408203125) -
                        0.07516433982238841793321881823430885560810565948486328125
                )
    g_m_large = 0.54116502161502622048061539317131973803043365478515625 * \
                tf.pow(alpha_indx4, -1.010274491769996618728555404231883585453033447265625)

    GO_Gamma_alpha_value4 = f_m_large + g_m_large * (x_indx4 - alpha_indx4)

    # Part 4_1
    # indx4_1 = tf.where(
    #     tf.logical_and(
    #         tf.logical_and(x > 1e-2, alpha > 500.),
    #         (x > (alpha + 2. * tf.log(alpha) * tf.sqrt(alpha)))
    #     )
    # )
    # alpha_indx4_1 = tf.gather_nd(alpha, indx4_1)
    # GO_Gamma_alpha_value4_1 = 1. / alpha_indx4_1 - tf.gather_nd(ex_gamma_xa, indx4_1) * \
    #     tf.gather_nd(psi_m_log, indx4_1) * tf.gather_nd(igamma_up_reg, indx4_1)

    # Summerize
    GO_Gamma_alpha = tf.sparse_to_dense(indx1, x.shape, GO_Gamma_alpha_value1) + \
                     tf.sparse_to_dense(indx2, x.shape, GO_Gamma_alpha_value2) + \
                     tf.sparse_to_dense(indx2_1, x.shape, GO_Gamma_alpha_value2_1) + \
                     tf.sparse_to_dense(indx3, x.shape, GO_Gamma_alpha_value3) + \
                     tf.sparse_to_dense(indx4, x.shape, GO_Gamma_alpha_value4)
    # + \
    # tf.sparse_to_dense(indx4_1, x.shape, GO_Gamma_alpha_value4_1)

    GO_Gamma_alpha = tf.cast(GO_Gamma_alpha, tf.float32)

    return GO_Gamma_alpha  # , x_l, x_r, f_l, f_m, f_r, g_l


# Recognition Model - q_W
def q_W(name, V, K, reuse=False):
    with tf.variable_scope('q_W' + name, reuse=reuse):
        W_aW = tf.get_variable("W_aW", [V, K], tf.float32,
                               tf.random_uniform_initializer(0.1, 10))
        RW_aW = max_m_grad(min_W_alpha_rate, W_aW)
        # RW_aW = tf.maximum(min_W_alpha_rate, W_aW)
        W_alpha = tf.nn.softplus(RW_aW)

        W_bW = tf.get_variable("W_bW", [V, K], tf.float32,
                               tf.random_uniform_initializer(0.1, 10))
        RW_bW = max_m_grad(min_W_beta_rate, W_bW)
        # RW_bW = tf.maximum(min_W_beta_rate, W_bW)
        W_beta = tf.nn.softplus(RW_bW)
        # W_beta = tf.nn.softplus(W_bW)
        # W_beta = min_m_grad(W_alpha / min_mean, W_beta)

        if MethodName == 'GO':
            W_hat1s = tf.random_gamma([1], tf.stop_gradient(W_alpha), 1.)
            W_hat1s = tf.maximum(min_W, tf.squeeze(W_hat1s, 0))
            Grad_W_alpha1 = GO_Gamma_v2(tf.stop_gradient(W_hat1s), tf.stop_gradient(W_alpha))
            W_hat1 = W_alpha * tf.stop_gradient(Grad_W_alpha1) - \
                     tf.stop_gradient(W_alpha * Grad_W_alpha1) + \
                     tf.stop_gradient(W_hat1s)
            W1_Fcorr = tf.zeros([1])

        if MethodName == 'GRep':
            posi0 = tf.polygamma(tf.constant(0,dtype=tf.float32),W_alpha)
            posi1 = tf.polygamma(tf.constant(1,dtype=tf.float32),W_alpha)
            W_hat1s = tf.random_gamma([1], tf.stop_gradient(W_alpha), 1.)
            W_hat1s = tf.maximum(min_W, tf.squeeze(W_hat1s, 0))
            epsilo = tf.stop_gradient( (tf.log(W_hat1s)-posi0)/tf.maximum((tf.pow(posi1,0.5)),1e-8) )
            log_W_hat1 = epsilo*tf.pow(posi1,0.5)+posi0
            W_hat1 = tf.exp( log_W_hat1 )
            W1_Fcorr = tf.reduce_sum(
                    - tf.lgamma(W_alpha) + (W_alpha-1.)*log_W_hat1 - W_hat1
                    + log_W_hat1 + 0.5 * tf.log( posi1 )
                    ) 

        if MethodName == 'RSVI':
            lambda_W1 = tf.squeeze(tf.random_gamma([1], W_alpha + Bf, 1.), 0)
            lambda_W1 = tf.stop_gradient(tf.maximum(min_W, lambda_W1))
            W_hat1, W1_Fcorr = reject_h_boosted(lambda_W1, W_alpha)

        W = W_hat1 / W_beta
        # W = tf.maximum(min_W, W)
        W = max_m_grad(min_W, W)

        return W, W_alpha, W_beta, W1_Fcorr


def log_gamma_minus(x, a1, b1, a2, b2):
    yout = tf.reduce_sum(
        a1 * tf.log(b1) - a2 * tf.log(b2)
        + tf.lgamma(a2) - tf.lgamma(a1)
        + (a1 - a2) * tf.log(x) - (b1 - b2) * x
    )
    return yout


def log_Gamma(x, a1, b1):
    yout = tf.reduce_sum(
        a1 * tf.log(b1) - tf.lgamma(a1)
        + (a1 - 1.) * tf.log(x) - b1 * x
    )
    return yout


def log_Poisson(x, lambda1):
    yout = tf.reduce_sum(
        x * tf.log(lambda1) - lambda1 - tf.lgamma(x + 1.)
    )
    return yout


# z ~ q(z|x) & W ~ q(W)
x = tf.placeholder(tf.float32, shape=[batch_size, K[0]])
z1, z_alpha1, z_beta1, z1_Fcorr = q_z_x('_z_1', x, K[1])
W1, W_alpha1, W_beta1, W1_Fcorr = q_W('_W_1', K[0], K[1])
if Layers >= 2:
    z2, z_alpha2, z_beta2, z2_Fcorr = q_z_x('_z_2', z1, K[2])
    W2, W_alpha2, W_beta2, W2_Fcorr = q_W('_W_2', K[1], K[2])
if Layers >= 3:
    z3, z_alpha3, z_beta3, z3_Fcorr = q_z_x('_z_3', z2, K[3])
    W3, W_alpha3, W_beta3, W3_Fcorr = q_W('_W_3', K[2], K[3])

# Calculate ELBO
# truncate Phitheta
ELBO_z_trunc = tf.placeholder(tf.float32, [5], name='ELBO_z_trunc')
ELBO_W_trunc = tf.placeholder(tf.float32, [5], name='ELBO_W_trunc')
ELBO_Wz_trunc = tf.placeholder(tf.float32, [5], name='ELBO_Wz_trunc')

# Layer 1
Wz1 = tf.matmul(z1, tf.transpose(W1))
Loglike = log_Poisson(x, Wz1) / (K[0] * batch_size)
Loglike_E = log_Poisson(x, Wz1) / (K[0] * batch_size)
E_recon1 = tf.reduce_mean(tf.abs(x - Wz1))
z1T = max_m_grad(ELBO_z_trunc[1], z1)
W1T = max_m_grad(ELBO_W_trunc[1], W1)
# z1T = tf.maximum(ELBO_z_trunc[1], z1)
# W1T = tf.maximum(ELBO_W_trunc[1], W1)
#z1T = z1
#W1T = W1
if Layers == 1:
    Log_pmq_z1 = log_gamma_minus(z1T, 0.1, 0.1,
                                 tf.stop_gradient(z_alpha1), tf.stop_gradient(z_beta1)
                                 ) / (K[0] * batch_size)
    Log_pmq_z1_E = log_gamma_minus(z1, 0.1, 0.1, z_alpha1, z_beta1) / (K[0] * batch_size)
else:
    Wz2 = tf.matmul(z2, tf.transpose(W2))
    # Wz2 = max_m_grad(ELBO_Wz_trunc[2], Wz2)
    Log_pmq_z1 = log_gamma_minus(z1T, alpha_z, alpha_z / Wz2,
                                 tf.stop_gradient(z_alpha1), tf.stop_gradient(z_beta1)
                                 ) / (K[0] * batch_size)
    Log_pmq_z1_E = log_gamma_minus(z1, alpha_z, alpha_z / Wz2, z_alpha1, z_beta1) / (K[0] * batch_size)
    E_recon2 = tf.reduce_mean(tf.abs(x - tf.matmul(Wz2, tf.transpose(W1))))
Log_pmq_W1 = log_gamma_minus(W1T, 0.1, 0.3,
                             tf.stop_gradient(W_alpha1), tf.stop_gradient(W_beta1)
                             ) / (K[0] * rho_mb * batch_size)
Log_pmq_W1_E = log_gamma_minus(W1, 0.1, 0.3, W_alpha1, W_beta1) / (K[0] * rho_mb * batch_size)
ELBO = Loglike + Log_pmq_z1 + Log_pmq_W1
ELBO_E = Loglike_E + Log_pmq_z1_E + Log_pmq_W1_E
# Layer 2
if Layers >= 2:
    z2T = max_m_grad(ELBO_z_trunc[2], z2)
    W2T = max_m_grad(ELBO_W_trunc[2], W2)
    # z2T = tf.maximum(ELBO_z_trunc[2], z2)
    # W2T = tf.maximum(ELBO_W_trunc[2], W2)
#    z2T = z2
#    W2T = W2
    if Layers == 2:
        Log_pmq_z2 = log_gamma_minus(z2T, 0.1, 0.1,
                                     tf.stop_gradient(z_alpha2), tf.stop_gradient(z_beta2)
                                     ) / (K[0] * batch_size)
        Log_pmq_z2_E = log_gamma_minus(z2, 0.1, 0.1, z_alpha2, z_beta2) / (K[0] * batch_size)
    else:
        Wz3 = tf.matmul(z3, tf.transpose(W3))
        # Wz3 = max_m_grad(ELBO_Wz_trunc[3], Wz3)
        Log_pmq_z2 = log_gamma_minus(z2T, alpha_z, alpha_z / Wz3,
                                     tf.stop_gradient(z_alpha2), tf.stop_gradient(z_beta2)
                                     ) / (K[0] * batch_size)
        Log_pmq_z2_E = log_gamma_minus(z2, alpha_z, alpha_z / Wz3, z_alpha2, z_beta2) / (K[0] * batch_size)
        E_recon3 = tf.reduce_mean(tf.abs(x - tf.matmul(tf.matmul(Wz3, tf.transpose(W2)), tf.transpose(W1))))
    Log_pmq_W2 = log_gamma_minus(W2T, 0.1, 0.3,
                                 tf.stop_gradient(W_alpha2), tf.stop_gradient(W_beta2)
                                 ) / (K[0] * rho_mb * batch_size)
    Log_pmq_W2_E = log_gamma_minus(W2, 0.1, 0.3, W_alpha2, W_beta2) / (K[0] * rho_mb * batch_size)
    ELBO = ELBO + Log_pmq_z2 + Log_pmq_W2
    ELBO_E = ELBO_E + Log_pmq_z2_E + Log_pmq_W2_E
# Layer 3
if Layers >= 3:
    # z3T = max_m_grad(ELBO_z_trunc[3], z3)
    # W3T = max_m_grad(ELBO_W_trunc[3], W3)
    # z3T = tf.maximum(ELBO_z_trunc[3], z3)
    # W3T = tf.maximum(ELBO_W_trunc[3], W3)
    z3T = z3
    W3T = W3
    if Layers == 3:
        Log_pmq_z3 = log_gamma_minus(z3T, 0.1, 0.1,
                                     tf.stop_gradient(z_alpha3), tf.stop_gradient(z_beta3)
                                     ) / (K[0] * batch_size)
        Log_pmq_z3_E = log_gamma_minus(z3, 0.1, 0.1, z_alpha3, z_beta3) / (K[0] * batch_size)
    else:
        Wz4 = tf.matmul(z4, tf.transpose(W4))
        # Wz4 = max_m_grad(ELBO_Wz_trunc[4], Wz4)
        Log_pmq_z3 = log_gamma_minus(z3T, alpha_z, alpha_z / Wz4,
                                     tf.stop_gradient(z_alpha3), tf.stop_gradient(z_beta3)
                                     ) / (K[0] * batch_size)
        Log_pmq_z3_E = log_gamma_minus(z3, alpha_z, alpha_z / Wz4, z_alpha3, z_beta3) / (K[0] * batch_size)
    Log_pmq_W3 = log_gamma_minus(W3T, 0.1, 0.3,
                                 tf.stop_gradient(W_alpha3), tf.stop_gradient(W_beta3)
                                 ) / (K[0] * rho_mb * batch_size)
    Log_pmq_W3_E = log_gamma_minus(W3, 0.1, 0.3, W_alpha3, W_beta3) / (K[0] * rho_mb * batch_size)
    ELBO = ELBO + Log_pmq_z3 + Log_pmq_W3
    ELBO_E = ELBO_E + Log_pmq_z3_E + Log_pmq_W3_E

if MethodName != 'GO':
#    ELBO = tf.stop_gradient(ELBO) * (z1_Fcorr + W1_Fcorr + z2_Fcorr + W2_Fcorr)\
#     - tf.stop_gradient(ELBO * (z1_Fcorr + W1_Fcorr + z2_Fcorr + W2_Fcorr)) \
#     + ELBO
     ELBO = tf.stop_gradient(ELBO) * (z1_Fcorr + W1_Fcorr) \
            - tf.stop_gradient(ELBO * (z1_Fcorr + W1_Fcorr)) \
            + ELBO


# Optimizer
optimizer_q_z1 = tf.train.AdamOptimizer(learning_rate=LR_q_z1)
q_z_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_z_x_z_1')
train_q_z1 = optimizer_q_z1.minimize(-ELBO, var_list=q_z_vars1)
optimizer_q_W1 = tf.train.AdamOptimizer(learning_rate=LR_q_W1)
q_W_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_W_W_1')
train_q_W1 = optimizer_q_W1.minimize(-ELBO, var_list=q_W_vars1)
if Layers >= 2:
    optimizer_q_z2 = tf.train.AdamOptimizer(learning_rate=LR_q_z2)
    q_z_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_z_x_z_2')
    train_q_z2 = optimizer_q_z2.minimize(-ELBO, var_list=q_z_vars2)
    optimizer_q_W2 = tf.train.AdamOptimizer(learning_rate=LR_q_W2)
    q_W_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_W_W_2')
    train_q_W2 = optimizer_q_W2.minimize(-ELBO, var_list=q_W_vars2)

init = tf.global_variables_initializer()

ELBOTrset = []
ELBOEvalset = []
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(init)

    for i in range(1, num_steps + 1):

        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.round(batch_x * 10.)

        ELBO_z_trunc_val = [0.1, 0.1, 0.1, 0.1, 0.1]
        ELBO_W_trunc_val = [0.01, 0.01, 0.01, 0.01, 0.01]  #
        ELBO_Wz_trunc_val = [0.3, 0.3, 0.3, 0.3, 0.3]  #

        if Layers == 1:
            _, _, ELBO1, ELBO_Eval1, \
            E_recon11, \
            z11, z_alpha11, z_beta11, \
            W11, W_alpha11, W_beta11, \
                = \
                sess.run([train_q_z1, train_q_W1, ELBO, ELBO_E,
                          E_recon1,
                          z1, z_alpha1, z_beta1,
                          W1, W_alpha1, W_beta1,
                          ],
                         feed_dict={x: batch_x,
                                    ELBO_z_trunc: ELBO_z_trunc_val,
                                    ELBO_W_trunc: ELBO_W_trunc_val,
                                    ELBO_Wz_trunc: ELBO_Wz_trunc_val,
                                    })
        if Layers == 2:
            _, _, _, _, ELBO1, ELBO_Eval1, \
            E_recon11, E_recon21, \
            Log_pmq_z11, Log_pmq_W11, \
            z11, z_alpha11, z_beta11, \
            W11, W_alpha11, W_beta11, \
            z21, z_alpha21, z_beta21, \
            W21, W_alpha21, W_beta21, \
                = \
                sess.run([train_q_z1, train_q_W1, train_q_z2, train_q_W2, ELBO, ELBO_E,
                          E_recon1, E_recon2,
                          Log_pmq_z1, Log_pmq_W1,
                          z1, z_alpha1, z_beta1,
                          W1, W_alpha1, W_beta1,
                          z2, z_alpha2, z_beta2,
                          W2, W_alpha2, W_beta2,
                          ],
                         feed_dict={x: batch_x,
                                    ELBO_z_trunc: ELBO_z_trunc_val,
                                    ELBO_W_trunc: ELBO_W_trunc_val,
                                    ELBO_Wz_trunc: ELBO_Wz_trunc_val,
                                    })
        if Layers == 3:
            _, _, _, _, _, _, ELBO1, ELBO_Eval1, \
            E_recon11, E_recon21, E_recon31, \
            z11, z_alpha11, z_beta11, \
            W11, W_alpha11, W_beta11, \
            z21, z_alpha21, z_beta21, \
            W21, W_alpha21, W_beta21, \
            z31, z_alpha31, z_beta31, \
            W31, W_alpha31, W_beta31, \
                = \
                sess.run([train_q_z1, train_q_W1, train_q_z2, train_q_W2, train_q_z3, train_q_W3, ELBO, ELBO_E,
                          E_recon1, E_recon2, E_recon3,
                          z1, z_alpha1, z_beta1,
                          W1, W_alpha1, W_beta1,
                          z2, z_alpha2, z_beta2,
                          W2, W_alpha2, W_beta2,
                          z3, z_alpha3, z_beta3,
                          W3, W_alpha3, W_beta3,
                          ],
                         feed_dict={x: batch_x,
                                    ELBO_z_trunc: ELBO_z_trunc_val,
                                    ELBO_W_trunc: ELBO_W_trunc_val,
                                    ELBO_Wz_trunc: ELBO_Wz_trunc_val,
                                    })

        ELBOTrset.append(ELBO1)
        ELBOEvalset.append(ELBO_Eval1)

        if i % 10 == 0:
            if Layers == 1:
                print('Step %5i: ELBO:[%.2f/%.2f], E_recon:[%.2f], '
                      'z1:[%.1e/%.1e/%.1e], W1:[%.1e/%.1e/%.1e], ' % (
                          i, ELBO_Eval1, ELBO1, E_recon11,
                          np.max(z11), np.max(z_alpha11), np.max(z_beta11),
                          np.max(W11), np.max(W_alpha11), np.max(W_beta11),
                      ))
#                if i % 200 == 0:
#                    f, a = plt.subplots(5, 6, sharex=True, sharey=True)
#                    for iii in range(5):
#                        for jjj in range(6):
#                            img = np.reshape(W11[:, iii * 6 + jjj], newshape=(28, 28))
#                            a[iii][jjj].imshow(img)
#                    f.show()
            if Layers == 2:
                print('Step %5i: ELBO:[%.2f/%.2f], E_recon:[%.2f/%.2f], '
                      'z1:[%.1e/%.1e/%.1e], W1:[%.1e/%.1e/%.1e], '
                      'z2:[%.1e/%.1e/%.1e], W2:[%.1e/%.1e/%.1e], ' % (
                          i, ELBO_Eval1, ELBO1, E_recon11, E_recon21,
                          np.max(z11), np.max(z_alpha11), np.max(z_beta11),
                          np.max(W11), np.max(W_alpha11), np.max(W_beta11),
                          np.max(z21), np.max(z_alpha21), np.max(z_beta21),
                          np.max(W21), np.max(W_alpha21), np.max(W_beta21),
                      ))
                # if i % 200 == 0:
                #     Dict = np.matmul(W11, W21)
                #     f, a = plt.subplots(5, 6, sharex=True, sharey=True)
                #     for iii in range(5):
                #         for jjj in range(6):
                #             img = np.reshape(Dict[:, iii * 6 + jjj], newshape=(28, 28))
                #             a[iii][jjj].imshow(img)
                #     f.show()
            if Layers == 3:
                print('Step %5i: ELBO:[%.2f/%.2f], E_recon:[%.2f/%.2f/%.2f], '
                      'z1:[%.1e/%.1e/%.1e], W1:[%.1e/%.1e/%.1e], '
                      'z2:[%.1e/%.1e/%.1e], W2:[%.1e/%.1e/%.1e], '
                      'z3:[%.1e/%.1e/%.1e], W3:[%.1e/%.1e/%.1e],' % (
                          i, ELBO_Eval1, ELBO1, E_recon11, E_recon21, E_recon31,
                          np.max(z11), np.max(z_alpha11), np.max(z_beta11),
                          np.max(W11), np.max(W_alpha11), np.max(W_beta11),
                          np.max(z21), np.max(z_alpha21), np.max(z_beta21),
                          np.max(W21), np.max(W_alpha21), np.max(W_beta21),
                          np.max(z31), np.max(z_alpha31), np.max(z_beta31),
                          np.max(W31), np.max(W_alpha31), np.max(W_beta31),
                      ))
#                if i % 200 == 0:
#                    Dict = np.matmul(W11, W21)
#                    f, a = plt.subplots(5, 6, sharex=True, sharey=True)
#                    for iii in range(5):
#                        for jjj in range(6):
#                            img = np.reshape(Dict[:, iii * 6 + jjj], newshape=(28, 28))
#                            a[iii][jjj].imshow(img)
#                    f.show()

        if i % 500 == 0:
            if Layers == 1:
                sio.savemat('./tmpmat1_' + MethodName + '.mat',
                            {'Iter': i,
                             'Layers': Layers,
                             'ELBOset': ELBOEvalset,
                             'ELBOTrset': ELBOTrset,
                             'x': batch_x,
                             'theta1': z11,
                             'theta_alpha1': z_alpha11,
                             'theta_beta1': z_beta11,
                             'phi1': W11,
                             'phi_alpha1': W_alpha11,
                             'phi_beta1': W_beta11,
                             'c2': 1.,
                             'c3': 1.,
                             })
            if Layers == 2:
                sio.savemat('./tmpmat2_' + MethodName + '.mat',
                            {'Iter': i,
                             'Layers': Layers,
                             'ELBOset': ELBOEvalset,
                             'ELBOTrset': ELBOTrset,
                             'x': batch_x,
                             'theta1': z11,
                             'theta_alpha1': z_alpha11,
                             'theta_beta1': z_beta11,
                             'phi1': W11,
                             'phi_alpha1': W_alpha11,
                             'phi_beta1': W_beta11,
                             'theta2': z21,
                             'theta_alpha2': z_alpha21,
                             'theta_beta2': z_beta21,
                             'phi2': W21,
                             'phi_alpha2': W_alpha21,
                             'phi_beta2': W_beta21,
                             'c2': 1.,
                             'c3': 1.,
                             })
            if Layers == 3:
                sio.savemat('./data/tmpmat3_' + MethodName + '.mat',
                            {'Iter': i,
                             'Layers': Layers,
                             'ELBOset': ELBOEvalset,
                             'ELBOTrset': ELBOTrset,
                             'x': batch_x,
                             'theta1': z11,
                             'theta_alpha1': z_alpha11,
                             'theta_beta1': z_beta11,
                             'phi1': W11,
                             'phi_alpha1': W_alpha11,
                             'phi_beta1': W_beta11,
                             'theta2': z21,
                             'theta_alpha2': z_alpha21,
                             'theta_beta2': z_beta21,
                             'phi2': W21,
                             'phi_alpha2': W_alpha21,
                             'phi_beta2': W_beta21,
                             'theta3': z31,
                             'theta_alpha3': z_alpha31,
                             'theta_beta3': z_beta31,
                             'phi3': W31,
                             'phi_alpha3': W_alpha31,
                             'phi_beta3': W_beta31,
                             'c2': 1.,
                             'c3': 1.,
                             })
