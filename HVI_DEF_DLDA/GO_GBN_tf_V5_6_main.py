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
num_steps = 200000
batch_size = 200
rho_mb = mnist.train.num_examples / batch_size

# Network Params
K = [784, 128, 64, 32]
h_dim = [0, 512, 256, 128]

Layers = 3
prior = 0.1
# eta_l = [0.1, 1./K[0], 1./K[1], 1./K[2]]
eta_l = [0.1, 0.1, 0.1, 0.1]
# eta_l = [0.5, 0.5, 0.5, 0.5]
# eta_l = [20., 20., 20., 20.]

is_trunc_ELBO = True
# trunc_theta_ELBO = 1e-25 # 1e-25 # 0.05
# trunc_c_ELBO = 1e-25 # 1e-25
# trunc_r_ELBO = 1e-25 # 1e-25
# trunc_phitheta_ELBO = 0.3 # 1e-20 # 0.25

Gama0 = 1.
c0 = 1.
e0 = 1.
f0 = 1.

LR_q_theta = 1e-3  # 1e-3  # 5e-4
LR_q_phi = 1.  # 1.  # 2e-1
LR_q_r = 1.  # 1.  # 1e-3

min_theta_alpha = 1e-10
min_theta_beta = 1e-10
min_phi_alpha = 1e-10

min_theta_alpha_rate = float(np.log(np.exp(min_theta_alpha) - 1.))
min_theta_beta_rate = float(np.log(np.exp(min_theta_beta) - 1.))
min_phi_alpha_rate = float(np.log(np.exp(min_phi_alpha) - 1.))

min_theta = 1e-25


# min_phi_RV = 1e-10 # 0.05


# Recognition Model - q_theta_x
def q_theta_x(name, lay, x, reuse=False):
    with tf.variable_scope('q_theta_x' + name, reuse=reuse):
        # if lay == 1:
        #     bb = sio.loadmat('C:/Users/yulai/Dropbox/[201709]Variance Reduction/Python Code/data/tmpmat5111.mat')
        #     theta1 = tf.constant(bb['theta1'], dtype=tf.float32)
        #     theta_alpha1 = tf.constant(bb['theta_alpha1'], dtype=tf.float32)
        #     theta_beta1 = tf.constant(bb['theta_beta1'], dtype=tf.float32)
        # else:
        # h1 = x
        h1 = tf.layers.batch_normalization(x)
        h1 = tf.nn.relu(
            tf.layers.dense(h1, units=h_dim[lay], kernel_initializer=tf.random_normal_initializer(0, 0.001)))
        h1 = tf.nn.relu(
            tf.layers.dense(h1, units=h_dim[lay], kernel_initializer=tf.random_normal_initializer(0, 0.001)))

        theta_Ralpha1 = tf.layers.dense(h1, units=K[lay], kernel_initializer=tf.random_normal_initializer(0, 0.001))
        theta_Ralpha1 = max_m_grad(min_theta_alpha_rate, theta_Ralpha1)
        theta_alpha1 = tf.nn.softplus(theta_Ralpha1)

        theta_Rbeta1 = tf.layers.dense(h1, units=K[lay], kernel_initializer=tf.random_normal_initializer(0, 0.001))
        theta_Rbeta1 = max_m_grad(min_theta_beta_rate, theta_Rbeta1)
        theta_beta1 = tf.nn.softplus(theta_Rbeta1)

        theta_hat1s = tf.random_gamma([1], tf.stop_gradient(theta_alpha1), 1.)
        theta_hat1s = tf.minimum(theta_hat1s,
                                 theta_alpha1 + tf.maximum(3., 2. * tf.log(theta_alpha1)) * tf.sqrt(theta_alpha1))
        theta_hat1s = tf.maximum(min_theta, tf.squeeze(theta_hat1s, 0))
        Grad_theta_alpha1 = GO_Gamma_v2(tf.stop_gradient(theta_hat1s), tf.stop_gradient(theta_alpha1))
        theta_hat1 = theta_alpha1 * tf.stop_gradient(Grad_theta_alpha1) - \
                     tf.stop_gradient(theta_alpha1 * Grad_theta_alpha1) + \
                     tf.stop_gradient(theta_hat1s)

        theta1 = theta_hat1 / theta_beta1

        # Next Layer - c_j
        # h2 = theta1
        h2 = tf.layers.batch_normalization(theta1)
        # h2 = tf.nn.relu(tf.layers.dense(h2, units=128, kernel_initializer=tf.random_normal_initializer(0, 0.001)))

        c_alpha2 = tf.layers.dense(h2, units=1, kernel_initializer=tf.random_normal_initializer(0, 0.001))
        c_alpha2 = tf.nn.softplus(max_m_grad(min_theta_alpha_rate, c_alpha2))

        c_beta2 = tf.layers.dense(h2, units=1, kernel_initializer=tf.random_normal_initializer(0, 0.001))
        c_beta2 = tf.nn.softplus(max_m_grad(min_theta_beta_rate, c_beta2))

        c_hat2s = tf.random_gamma([1], tf.stop_gradient(c_alpha2), 1.)
        c_hat2s = tf.minimum(c_hat2s, c_alpha2 + tf.maximum(3., 2. * tf.log(c_alpha2)) * tf.sqrt(c_alpha2))
        c_hat2s = tf.maximum(min_theta, tf.squeeze(c_hat2s, 0))
        Grad_c_alpha2 = GO_Gamma_v2(tf.stop_gradient(c_hat2s), tf.stop_gradient(c_alpha2))
        c_hat2 = c_alpha2 * tf.stop_gradient(Grad_c_alpha2) - \
                 tf.stop_gradient(c_alpha2 * Grad_c_alpha2) + \
                 tf.stop_gradient(c_hat2s)

        c2 = 0.1 + c_hat2 / c_beta2
        # c2 = max_m_grad(0.3, c2)

        return theta1, theta_alpha1, theta_beta1, c2, c_alpha2, c_beta2


# Recognition Model - q_Phi
def q_psi_phi(name, V, K, reuse=False):
    with tf.variable_scope('q_psi_phi' + name, reuse=reuse):
        W_phi = tf.get_variable("W_Phi", [V, K], tf.float32,
                                tf.random_normal_initializer(0, 0.1))
        phi_alpha = tf.nn.softplus(max_m_grad(min_phi_alpha_rate, W_phi))

        psi_phis = tf.random_gamma([1], tf.stop_gradient(phi_alpha), 1.)
        psi_phis = tf.minimum(psi_phis, phi_alpha + tf.maximum(3., 2. * tf.log(phi_alpha)) * tf.sqrt(phi_alpha))
        psi_phis = max_m_grad(min_theta, tf.squeeze(psi_phis, 0))
        Grad_phi_alpha = GO_Gamma_v2(tf.stop_gradient(psi_phis), tf.stop_gradient(phi_alpha))
        psi_phi = phi_alpha * tf.stop_gradient(Grad_phi_alpha) \
                  - tf.stop_gradient(phi_alpha * Grad_phi_alpha) \
                  + tf.stop_gradient(psi_phis)

        phi = psi_phi / tf.reduce_sum(psi_phi, 0, keepdims=True)
        # phi = map_Dir_64(phi, min_phi_RV / V)

        return phi, phi_alpha


def q_r(reuse=False):
    with tf.variable_scope('q_r', reuse=reuse):
        r_alpha_W = tf.get_variable("r_alpha_W", [K[Layers], 1], tf.float32,
                                    tf.random_normal_initializer(0., 0.001))
        r_alpha = tf.nn.softplus(max_m_grad(min_theta_alpha_rate, r_alpha_W))

        r_beta_W = tf.get_variable("r_beta_W", [1, 1], tf.float32,
                                   tf.random_normal_initializer(0., 0.001))
        r_beta = tf.nn.softplus(max_m_grad(min_theta_beta_rate, r_beta_W))

        r_hats = tf.random_gamma([1], tf.stop_gradient(r_alpha), 1.)
        r_hats = tf.minimum(r_hats, r_alpha + tf.maximum(3., 2. * tf.log(r_alpha)) * tf.sqrt(r_alpha))
        r_hats = tf.maximum(min_theta, tf.squeeze(r_hats, 0))
        Grad_r_alpha = GO_Gamma_v2(tf.stop_gradient(r_hats), tf.stop_gradient(r_alpha))
        r_hat = r_alpha * tf.stop_gradient(Grad_r_alpha) - \
                tf.stop_gradient(r_alpha * Grad_r_alpha) + \
                tf.stop_gradient(r_hats)

        r = r_hat / r_beta

        return r, r_alpha, r_beta


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


def log_gamma_minus(x, a1, b1, a2, b2):
    yout = tf.reduce_sum(
        a1 * tf.log(b1) - a2 * tf.log(b2)
        + tf.lgamma(a2) - tf.lgamma(a1)
        + (a1 - a2) * tf.log(x) - (b1 - b2) * x
    )
    return yout


def log_gamma(x, a1, b1):
    yout = tf.reduce_sum(
        a1 * tf.log(b1) - tf.lgamma(a1)
        + (a1 - 1.) * tf.log(x) - b1 * x
    )
    return yout


# two log-Dirichlet pdfs minus
def log_dirichlet_minus(x, eta1, eta2):  # dimension mismatch problem unsolved
    V = float(x.shape[0].value)
    yout = tf.reduce_sum(
        1. / V * (tf.lgamma(V * eta1) - tf.lgamma(tf.reduce_sum(eta2, 0, keepdims=True)))
        + tf.lgamma(eta2) - tf.lgamma(eta1)
        + (eta1 - eta2) * tf.log(x)
    )
    return yout


# theta ~ q(theta)
x = tf.placeholder(tf.float32, shape=[batch_size, K[0]])
theta1, theta_alpha1, theta_beta1, c2, c_alpha2, c_beta2 = q_theta_x('theta_1', 1, x)
phi1, phi_alpha1 = q_psi_phi('phi_1', K[0], K[1])
theta1T = max_m_grad(1e-5, theta1)
c2T = max_m_grad(1e-5, c2)
if Layers >= 2:
    theta2, theta_alpha2, theta_beta2, c3, c_alpha3, c_beta3 = q_theta_x('theta_2', 2, theta1)
    phi2, phi_alpha2 = q_psi_phi('phi_2', K[1], K[2])
    theta2T = max_m_grad(1e-5, theta2)
    c3T = max_m_grad(1e-5, c3)
if Layers >= 3:
    theta3, theta_alpha3, theta_beta3, c4, c_alpha4, c_beta4 = q_theta_x('theta_3', 3, theta2)
    phi3, phi_alpha3 = q_psi_phi('phi_3', K[2], K[3])
    theta3T = max_m_grad(1e-5, theta3)
    c4T = max_m_grad(1e-5, c4)
# r, r_alpha, r_beta = q_r()
r = tf.ones([K[Layers], 1]) * 0.2  # Gama0 / K[Layers]
r_alpha = tf.ones([1])
r_beta = tf.ones([1])

phi_theta1 = tf.matmul(theta1, tf.transpose(phi1))
E_recon1 = tf.reduce_mean(tf.abs(x - phi_theta1))
if Layers >= 2:
    phi_theta2 = tf.matmul(theta2, tf.transpose(phi2))
    E_recon2 = tf.reduce_mean(tf.abs(x - tf.matmul(phi_theta2 / c2, tf.transpose(phi1))))
    # phi_theta2T = max_m_grad(trunc_phitheta_ELBO, phi_theta2)
    phi_theta2T = max_m_grad(0.2, phi_theta2)
if Layers >= 3:
    phi_theta3 = tf.matmul(theta3, tf.transpose(phi3))
    E_recon3 = tf.reduce_mean(
        tf.abs(x - tf.matmul(tf.matmul(phi_theta3 / c3, tf.transpose(phi2)) / c2, tf.transpose(phi1))))
    # phi_theta3T = max_m_grad(trunc_phitheta_ELBO, phi_theta3)
    phi_theta3T = max_m_grad(0.2, phi_theta3)

# Stop gradient vals
theta1_s = tf.stop_gradient(theta1)
theta1T_s = tf.stop_gradient(theta1T)
theta_alpha1_s = tf.stop_gradient(theta_alpha1)
theta_beta1_s = tf.stop_gradient(theta_beta1)
c2_s = tf.stop_gradient(c2)

c_alpha2_s = tf.stop_gradient(c_alpha2)
c_beta2_s = tf.stop_gradient(c_beta2)
phi1_s = tf.stop_gradient(phi1)
phi_alpha1_s = tf.stop_gradient(phi_alpha1)
if Layers >= 2:
    theta2_s = tf.stop_gradient(theta2)
    theta2T_s = tf.stop_gradient(theta2T)
    theta_alpha2_s = tf.stop_gradient(theta_alpha2)
    theta_beta2_s = tf.stop_gradient(theta_beta2)
    c3_s = tf.stop_gradient(c3)
    c_alpha3_s = tf.stop_gradient(c_alpha3)
    c_beta3_s = tf.stop_gradient(c_beta3)
    phi2_s = tf.stop_gradient(phi2)
    phi_alpha2_s = tf.stop_gradient(phi_alpha2)
if Layers >= 3:
    theta3_s = tf.stop_gradient(theta3)
    theta3T_s = tf.stop_gradient(theta3T)
    theta_alpha3_s = tf.stop_gradient(theta_alpha3)
    theta_beta3_s = tf.stop_gradient(theta_beta3)
    c4_s = tf.stop_gradient(c4)
    c_alpha4_s = tf.stop_gradient(c_alpha4)
    c_beta4_s = tf.stop_gradient(c_beta4)
    phi3_s = tf.stop_gradient(phi3)
    phi_alpha3_s = tf.stop_gradient(phi_alpha3)
r_s = tf.stop_gradient(r)
r_alpha_s = tf.stop_gradient(r_alpha)
r_beta_s = tf.stop_gradient(r_beta)

# Calculate ELBO
Loglike = tf.reduce_sum(x * tf.log(phi_theta1) - phi_theta1 - tf.lgamma(x + 1.)) / (K[0] * batch_size)
# Layer 1
if Layers == 1:
    pta_phi_theta2 = tf.transpose(r)
    pta_phi_theta2T = tf.transpose(r)
else:
    pta_phi_theta2 = phi_theta2
    pta_phi_theta2T = phi_theta2T
pta_phi_theta2_s = tf.stop_gradient(pta_phi_theta2)
pta_phi_theta2T_s = tf.stop_gradient(pta_phi_theta2T)

ELBO = Loglike + log_gamma_minus(theta1T, pta_phi_theta2T_s, c2_s, theta_alpha1_s, theta_beta1_s) / (K[0] * batch_size)
ELBO = ELBO + log_dirichlet_minus(phi1, eta_l[1], phi_alpha1_s) / (K[0] * rho_mb * batch_size)
ELBO = ELBO + log_gamma_minus(theta1T_s, pta_phi_theta2T_s, c2, theta_alpha1_s, theta_beta1_s) / (K[0] * batch_size) \
       + prior * log_gamma_minus(c2, e0, f0, c_alpha2_s, c_beta2_s) / (K[0] * batch_size)

# Layer 2
if Layers >= 2:
    if Layers == 2:
        pta_phi_theta3 = tf.transpose(r)
        pta_phi_theta3T = tf.transpose(r)
    else:
        pta_phi_theta3 = phi_theta3
        pta_phi_theta3T = phi_theta3T
    pta_phi_theta3_s = tf.stop_gradient(pta_phi_theta3)
    pta_phi_theta3T_s = tf.stop_gradient(pta_phi_theta3T)

    ELBO = ELBO + log_gamma_minus(theta1T_s, pta_phi_theta2T, c2_s, theta_alpha1_s, theta_beta1_s) / (K[0] * batch_size)
    ELBO = ELBO + prior * log_gamma_minus(theta2T, pta_phi_theta3T_s, c3_s, theta_alpha2_s, theta_beta2_s) / (
                K[0] * batch_size)
    ELBO = ELBO + prior * log_dirichlet_minus(phi2, eta_l[2], phi_alpha2_s) / (K[0] * rho_mb * batch_size)
    ELBO = ELBO + log_gamma_minus(theta2T_s, pta_phi_theta3T_s, c3, theta_alpha2_s, theta_beta2_s) / (K[0] * batch_size) \
           + prior * log_gamma_minus(c3, e0, f0, c_alpha3_s, c_beta3_s) / (K[0] * batch_size)

    loglike_t1 = log_gamma(theta1_s, pta_phi_theta2, c2_s) / (K[0] * batch_size)
    ELBOa1 = log_gamma_minus(theta1T_s, pta_phi_theta2T, c2_s, theta_alpha1_s, theta_beta1_s) / (K[0] * batch_size)
    ELBOa2 = log_gamma_minus(theta2T, pta_phi_theta3T_s, c3_s, theta_alpha2_s, theta_beta2_s) / (K[0] * batch_size)
    ELBOa3 = log_dirichlet_minus(phi2, eta_l[2], phi_alpha2_s) / (K[0] * rho_mb * batch_size)
    ELBOa4 = log_gamma_minus(theta2T_s, pta_phi_theta3T_s, c3T, theta_alpha2_s, theta_beta2_s) / (K[0] * batch_size)
    ELBOa5 = log_gamma_minus(c3T, e0, f0, c_alpha3_s, c_beta3_s) / (K[0] * batch_size)
# Layer 3
if Layers >= 3:
    if Layers == 3:
        pta_phi_theta4 = tf.transpose(r)
        pta_phi_theta4T = tf.transpose(r)
    # else:
    #     pta_phi_theta4 = phi_theta4
    #     pta_phi_theta4T = phi_theta4T
    pta_phi_theta4_s = tf.stop_gradient(pta_phi_theta4)
    pta_phi_theta4T_s = tf.stop_gradient(pta_phi_theta4T)

    ELBO = ELBO + log_gamma_minus(theta2T_s, pta_phi_theta3T, c3_s, theta_alpha2_s, theta_beta2_s) / (K[0] * batch_size)
    ELBO = ELBO + prior * log_gamma_minus(theta3T, pta_phi_theta4T_s, c4_s, theta_alpha3_s, theta_beta3_s) / (
                K[0] * batch_size)
    ELBO = ELBO + prior * log_dirichlet_minus(phi3, eta_l[3], phi_alpha3_s) / (K[0] * rho_mb * batch_size)
    ELBO = ELBO + log_gamma_minus(theta3T_s, pta_phi_theta4T_s, c4, theta_alpha3_s, theta_beta3_s) / (K[0] * batch_size) \
           + prior * log_gamma_minus(c4, e0, f0, c_alpha4_s, c_beta4_s) / (K[0] * batch_size)

    loglike_t2 = log_gamma(theta2_s, pta_phi_theta3, c3_s) / (K[0] * batch_size)

# r
if Layers == 1:
    #     ELBO = ELBO + log_gamma_minus(theta1_s, tf.transpose(r), c2_s, theta_alpha1_s, theta_beta1_s) / (K[0] * batch_size)
    loglike_t1 = log_gamma(theta1_s, tf.transpose(r), c2_s)
elif Layers == 2:
    #     ELBO = ELBO + log_gamma_minus(theta2_s, tf.transpose(r), c3_s, theta_alpha2_s, theta_beta2_s) / (K[0] * batch_size)
    loglike_t2 = log_gamma(theta2_s, tf.transpose(r), c3_s)
elif Layers == 3:
    #     ELBO = ELBO + log_gamma_minus(theta3_s, tf.transpose(r), c4_s, theta_alpha3_s, theta_beta3_s) / (K[0] * batch_size)
    loglike_t3 = log_gamma(theta3_s, tf.transpose(r), c4_s)
# ELBO = ELBO + log_gamma_minus(r, Gama0 / K[Layers], c0, r_alpha_s, r_beta_s) / (K[0] * rho_mb * batch_size)

# Optimizer
optimizer_q_theta = tf.train.AdamOptimizer(learning_rate=LR_q_theta)
optimizer_q_phi = tf.train.AdamOptimizer(learning_rate=LR_q_phi)
optimizer_q_r = tf.train.AdamOptimizer(learning_rate=LR_q_r)

q_theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_theta_x')
q_phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_psi_phi')
q_r_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_r')

train_q_theta = optimizer_q_theta.minimize(-ELBO, var_list=q_theta_vars)
train_q_phi = optimizer_q_phi.minimize(-ELBO, var_list=q_phi_vars)
# train_q_r = optimizer_q_r.minimize(-ELBO, var_list=q_r_vars)
# train_q_phi = tf.zeros([1])
train_q_r = tf.zeros([1])

init = tf.global_variables_initializer()

ELBOset = []
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(init)

    # batch_x, _ = mnist.train.next_batch(batch_size)
    # batch_x = np.round(batch_x * 10.)
    # bb = sio.loadmat('C:/Users/yulai/Dropbox/[201709]Variance Reduction/Python Code/data/tmpmat5111.mat')
    # batch_x = bb['x']

    for i in range(1, num_steps + 1):

        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.round(batch_x * 10.)

        if Layers == 1:
            _, _, _, ELBO1, E_recon11, \
            theta11, theta_alpha11, theta_beta11, c21, c_alpha21, c_beta21, \
            phi11, phi_alpha11, \
            r1, r_alpha1, r_beta1 = \
                sess.run([train_q_theta, train_q_phi, train_q_r, ELBO, E_recon1,
                          theta1, theta_alpha1, theta_beta1, c2, c_alpha2, c_beta2,
                          phi1, phi_alpha1,
                          r, r_alpha, r_beta,
                          ],
                         feed_dict={x: batch_x,
                                    })
        if Layers == 2:
            _, _, _, ELBO1, E_recon11, E_recon21, \
            ELBOa11, ELBOa21, ELBOa31, ELBOa41, ELBOa51, phi_theta21, \
            loglike1, loglike_t11, loglike_t21, \
            theta11, theta_alpha11, theta_beta11, c21, c_alpha21, c_beta21, \
            theta21, theta_alpha21, theta_beta21, c31, c_alpha31, c_beta31, \
            phi11, phi_alpha11, phi21, phi_alpha21, \
            r1, r_alpha1, r_beta1 = \
                sess.run([train_q_theta, train_q_phi, train_q_r, ELBO, E_recon1, E_recon2,
                          ELBOa1, ELBOa2, ELBOa3, ELBOa4, ELBOa5, phi_theta2,
                          Loglike, loglike_t1, loglike_t2,
                          theta1, theta_alpha1, theta_beta1, c2, c_alpha2, c_beta2,
                          theta2, theta_alpha2, theta_beta2, c3, c_alpha3, c_beta3,
                          phi1, phi_alpha1, phi2, phi_alpha2,
                          r, r_alpha, r_beta,
                          ],
                         feed_dict={x: batch_x,
                                    })
        if Layers == 3:
            _, _, _, ELBO1, E_recon11, E_recon21, E_recon31, \
            loglike1, loglike_t11, loglike_t21, loglike_t31, \
            theta11, theta_alpha11, theta_beta11, c21, c_alpha21, c_beta21, \
            theta21, theta_alpha21, theta_beta21, c31, c_alpha31, c_beta31, \
            theta31, theta_alpha31, theta_beta31, c41, c_alpha41, c_beta41, \
            phi11, phi_alpha11, phi21, phi_alpha21, phi31, phi_alpha31, \
            r1, r_alpha1, r_beta1 = \
                sess.run([train_q_theta, train_q_phi, train_q_r, ELBO, E_recon1, E_recon2, E_recon3,
                          Loglike, loglike_t1, loglike_t2, loglike_t3,
                          theta1, theta_alpha1, theta_beta1, c2, c_alpha2, c_beta2,
                          theta2, theta_alpha2, theta_beta2, c3, c_alpha3, c_beta3,
                          theta3, theta_alpha3, theta_beta3, c4, c_alpha4, c_beta4,
                          phi1, phi_alpha1, phi2, phi_alpha2, phi3, phi_alpha3,
                          r, r_alpha, r_beta,
                          ],
                         feed_dict={x: batch_x,
                                    })

        ELBOset.append(ELBO1)

        if i % 10 == 0:
            if Layers == 1:
                print('Step %5i: ELBO:[%.2f], E_recon:[%.2f], r:[%.1e/%.1e/%.1e], '
                      't1:[%.1e/%.1e/%.1e], c2:[%.1e/%.1e/%.1e/%.1e]，'
                      'phi1:[%.1e/%.1e/%.1e/%.1e], ' % (
                          i, ELBO1, E_recon11,
                          np.max(r1), np.max(r_alpha1), np.max(r_beta1),
                          np.max(theta11), np.max(theta_alpha11), np.max(theta_beta11),
                          np.min(c21), np.max(c21), np.max(c_alpha21), np.max(c_beta21),
                          np.min(phi11), np.max(phi11), np.min(phi_alpha11), np.max(phi_alpha11),
                      ))
            if Layers == 2:
                print('Step %5i: ELBO:[%.2f], E_recon:[%.2f/%.2f], '
                      '[%.2e/%.2e/%.2e], '
                      'r:[%.1e/%.1e/%.1e], '
                      't1:[%.1e/%.1e/%.1e], c2:[%.1e/%.1e/%.1e/%.1e], '
                      't2:[%.1e/%.1e/%.1e], c3:[%.1e/%.1e/%.1e/%.1e], '
                      'phi1:[%.1e/%.1e/%.1e/%.1e], phi2:[%.1e/%.1e/%.1e/%.1e], ' % (
                          i, ELBO1, E_recon11, E_recon21,
                          loglike1, loglike_t11, loglike_t21,
                          np.max(r1), np.max(r_alpha1), np.max(r_beta1),
                          np.max(theta11), np.max(theta_alpha11), np.max(theta_beta11),
                          np.min(c21), np.max(c21), np.max(c_alpha21), np.max(c_beta21),
                          np.max(theta21), np.max(theta_alpha21), np.max(theta_beta21),
                          np.min(c31), np.max(c31), np.max(c_alpha31), np.max(c_beta31),
                          np.min(phi11), np.max(phi11), np.min(phi_alpha11), np.max(phi_alpha11),
                          np.min(phi21), np.max(phi21), np.min(phi_alpha21), np.max(phi_alpha21),
                      ))
            if Layers == 3:
                # print(grad_norm_t1, grad_norm_p1)
                print('Step %5i: ELBO:[%.2f], E_recon:[%.2f/%.2f/%.2f], '
                      '[%.2e/%.2e/%.2e/%.2e], '
                      'r:[%.1e/%.1e/%.1e], '
                      't1:[%.1e/%.1e/%.1e], c2:[%.1e/%.1e/%.1e/%.1e], '
                      't2:[%.1e/%.1e/%.1e], c3:[%.1e/%.1e/%.1e/%.1e], '
                      't3:[%.1e/%.1e/%.1e], c4:[%.1e/%.1e/%.1e/%.1e], '
                      'phi:[%.1e/%.1e/%.1e], ' % (
                          i, ELBO1, E_recon11, E_recon21, E_recon31,
                          loglike1, loglike_t11, loglike_t21, loglike_t31,
                          np.max(r1), np.max(r_alpha1), np.max(r_beta1),
                          np.max(theta11), np.max(theta_alpha11), np.max(theta_beta11),
                          np.min(c21), np.max(c21), np.max(c_alpha21), np.max(c_beta21),
                          np.max(theta21), np.max(theta_alpha21), np.max(theta_beta21),
                          np.min(c31), np.max(c31), np.max(c_alpha31), np.max(c_beta31),
                          np.max(theta31), np.max(theta_alpha31), np.max(theta_beta31),
                          np.min(c41), np.max(c41), np.max(c_alpha41), np.max(c_beta41),
                          np.max(phi_alpha11), np.max(phi_alpha21), np.max(phi_alpha31),
                      ))

        if i % 500 == 0:
            if Layers == 1:
                sio.savemat('./data/tmpmat5.mat', {'Iter': i,
                                                   'Layers': Layers,
                                                   'ELBOset': ELBOset,
                                                   'x': batch_x,
                                                   'theta1': theta11,
                                                   'theta_alpha1': theta_alpha11,
                                                   'theta_beta1': theta_beta11,
                                                   'phi1': phi11,
                                                   'c2': c21,
                                                   })
            if Layers == 2:
                sio.savemat('./data/tmpmat5.mat', {'Iter': i,
                                                   'Layers': Layers,
                                                   'ELBOset': ELBOset,
                                                   'x': batch_x,
                                                   'theta1': theta11,
                                                   'theta_alpha1': theta_alpha11,
                                                   'theta_beta1': theta_beta11,
                                                   'theta2': theta21,
                                                   'theta_alpha2': theta_alpha21,
                                                   'theta_beta2': theta_beta21,
                                                   'c2': c21,
                                                   'phi1': phi11,
                                                   'phi2': phi21,
                                                   'phi_theta2': phi_theta21,
                                                   })
            if Layers == 3:
                sio.savemat('./data/tmpmat5.mat', {'Iter': i,
                                              'Layers': Layers,
                                              'ELBOset': ELBOset,
                                              'x': batch_x,
                                              'theta1': theta11,
                                              'theta_alpha1': theta_alpha11,
                                              'theta_beta1': theta_beta11,
                                              'theta2': theta21,
                                              'theta_alpha2': theta_alpha21,
                                              'theta_beta2': theta_beta21,
                                              'theta3': theta31,
                                              'theta_alpha3': theta_alpha31,
                                              'theta_beta3': theta_beta31,
                                              'c2': c21,
                                              'c3': c31,
                                              'c4': c41,
                                              'phi1': phi11,
                                              'phi2': phi21,
                                              'phi3': phi31,
                                              'r': r1
                                              })
                if (i + 1) % 20000 == 0:
                    sio.savemat('./GBN_l3_all5_%i.mat' % (i + 1), {'x': batch_x,
                                                                   'theta1': theta11,
                                                                   'theta_alpha1': theta_alpha11,
                                                                   'theta_beta1': theta_beta11,
                                                                   'c2': c21,
                                                                   'c_alpha2': c_alpha21,
                                                                   'c_beta2': c_beta21,
                                                                   'theta2': theta21,
                                                                   'theta_alpha2': theta_alpha21,
                                                                   'theta_beta2': theta_beta21,
                                                                   'c3': c31,
                                                                   'c_alpha3': c_alpha31,
                                                                   'c_beta3': c_beta31,
                                                                   'theta3': theta31,
                                                                   'theta_alpha3': theta_alpha31,
                                                                   'theta_beta3': theta_beta31,
                                                                   'c4': c41,
                                                                   'phi_alpha1': phi_alpha11,
                                                                   'phi1': phi11,
                                                                   'phi_alpha2': phi_alpha21,
                                                                   'phi2': phi21,
                                                                   'phi_alpha3': phi_alpha31,
                                                                   'phi3': phi31,
                                                                   'r': r1,
                                                                   'r_alpha': r_alpha1,
                                                                   'r_beta': r_beta1,
                                                                   })
