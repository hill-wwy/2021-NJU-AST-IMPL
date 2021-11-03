import os
import numpy as np
from sklearn import manifold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib.pyplot as plt
from keras.layers import Input
from core.util import print_accuracy,LearningHandler
from core import Conv
import scipy.io as scio
import tensorflow as tf
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from keras.utils import to_categorical
import imageio
from keras.models import Model
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds
# import cv2


def run_net(data, params):

    x_train_unlabeled, y_train_unlabeled, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']






    inputs_vae = Input(shape=(params['img_dim'],params['img_dim'],1), name='inputs_vae')
    # Tensor("inputs_vae:0", shape=(?, 28, 28, 1), dtype=float32)
    ConvAE = Conv.ConvAE(inputs_vae,params)
    ConvAE.vae.load_weights('MNIST_64.h5')
    ConvAE.Advsior.load_weights('MNIST_ADV_64.h5')
    lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                         patience=params['spec_patience'])  # 班级管理学习率调度和早期停止标准





    lh.on_train_begin() #在训练开始时初始化参数(这样可以在多次训练中重用该类)
    # one_hots = to_categorical(y_val,10)
    losses_vae = np.empty((1000,))
    acc = np.empty((1000,))
    losse = np.empty((1000,))
    nmi1 = np.empty((1000,))
    noise = 1*np.random.rand(13000,params['latent_dim'])
    noise1 = 1 * np.random.rand(13000, params['n_clusters'])




    for i in range(2):#原来1000         training_epochs



        #数据加入，前向传播
        x_val_t = ConvAE.encoder.predict(x_val)
        # scale = conv1.get_scale(x_val_y, 1000, params['scale_nbr'])
        x_val_t1 = ConvAE.Advsior.predict(x_val)  #
        # q= target_distribution(x_val_y)
        x_sp = ConvAE.classfier.predict(x_val_t)
        y_sp = x_sp.argmax(axis=1)
        x_val_y = ConvAE.classfier.predict(x_val_t1+x_val_t)
        y_sp_1 = x_val_y.argmax(axis=1)


        x_val_1 = ConvAE.decoder.predict(x_val_t)
        x_val_2 = ConvAE.decoder.predict(x_val_t1+x_val_t)



        accuracy = print_accuracy(y_sp, y_val, params['n_clusters'])
        nmi1[i] = accuracy
        accuracy = print_accuracy(y_sp_1, y_val, params['n_clusters'])

        #fit，调整  主要根据内部已经计算出的两个变量，*******************************noise?**
        losses_vae[i] = ConvAE.train_defense(x_val,noise,noise1,params['batch_size'])
        #create handler for early stopping and learning rate scheduling  调整todo
        # print("1Z Epoch: {}, loss={:2f},D = {}".format(i, losses_vae[i],M))
        # print("1Z Epoch: {}, loss={:2f},D = {}".format(i, losses_vae[i],M))
        acc[i] = accuracy

        # nmi1[i] = nmi(y_sp, y_val)
        print('NMI: ' + str(np.round(nmi(y_sp, y_val), 4)))

        print('NMI: ' + str(np.round(nmi(y_sp_1, y_val), 4)))

        if i>1:
            if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                print('STOPPING EARLY')
                break

    #可视化
    x_val_t = ConvAE.encoder.predict(x_val)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(x_val_t)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()

    print("finished training")

    x_val_y = ConvAE.vae.predict(x_val)[2]
    y_sp = x_val_y.argmax(axis=1)

    print_accuracy(y_sp, y_val, params['n_clusters'])

    nmi_score1 = nmi(y_sp, y_val)
    print('NMI: ' + str(np.round(nmi_score1, 4)))





def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
                             # 目标分布P，增强了软标签Q的识别能力
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T






