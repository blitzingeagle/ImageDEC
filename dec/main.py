from __future__ import division
from __future__ import print_function

import dec
import pretrain
import imageset_utils as imgutils

import cv2
import numpy as np

import os
import os.path as path
import shutil
import time


def load_mnist(root, training):
    if training:
        data = 'train-images-idx3-ubyte'
        label = 'train-labels-idx1-ubyte'
        N = 60000
    else:
        data = 't10k-images-idx3-ubyte'
        label = 't10k-labels-idx1-ubyte'
        N = 10000
    with open(root+data, 'rb') as fin:
        fin.seek(16, os.SEEK_SET)
        X = np.fromfile(fin, dtype=np.uint8).reshape((N,28*28))
        # for x in X:
        #     x = x.reshape((28,28))
        #     cv2.imshow("mnist", x)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
    with open(root+label, 'rb') as fin:
        fin.seek(8, os.SEEK_SET)
        Y = np.fromfile(fin, dtype=np.uint8)
    return X, Y

def make_mnist_data():
    X, Y = load_mnist('../mnist/', True)
    X = X.astype(np.float64)*0.02
    dec.write_db(X, Y, 'mnist_train')

    X_, Y_ = dec.read_db('mnist_train', True)
    assert np.abs((X - X_)).mean() < 1e-5
    assert (Y != Y_).sum() == 0

    X2, Y2 = load_mnist('../mnist/', False)
    X2 = X2.astype(np.float64)*0.02
    dec.write_db(X2, Y2, 'mnist_test')

    X3 = np.concatenate((X,X2), axis=0)
    Y3 = np.concatenate((Y,Y2), axis=0)
    dec.write_db(X3,Y3, 'mnist_total')


def DisKmeans():
    input_dir = "../images"
    imageset = resize_images(load_imageset(input_dir, cv2.IMREAD_GRAYSCALE), (50, 50))
    data = columnize(imageset)
    print("Data Loaded")

    db = "image"
    update_interval=160

    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    from sklearn.lda import LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.spatial.distance import cdist
    import cPickle
    from scipy.io import loadmat

    N_class = 5
    batch_size = 100
    train_batch_size = 256
    # X, Y = dec.read_db(db+'_total', True)
    # X = np.asarray(X, dtype=np.float64)
    # Y = np.asarray(np.squeeze(Y), dtype = np.int32)
    X = np.asarray(data)
    Y = np.zeros(len(data))
    N = X.shape[0]
    # img = np.clip((X/0.02), 0, 255).astype(np.uint8).reshape((N, 28, 28, 1))

    print(X.shape)
    print(Y.shape)
    print(N)

    tmm_alpha = 1.0
    total_iters = (N-1)/train_batch_size+1
    if not update_interval:
      update_interval = total_iters
    Y_pred = np.zeros((Y.shape[0]))
    iters = 0
    seek = 0
    dim = 10

    acc_list = []

    output_dir = "output"
    if path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    for class_idx in range(N_class):
        group_dir = path.join(output_dir, "group%04d" % class_idx)
        os.makedirs(group_dir)

    while iters < 1:
        # raw_input("Iteration %d" % iters)
        dec.write_net(db, dim, N_class, "'{:08}'".format(0))
        if iters == 0:
            dec.write_db(np.zeros((N,N_class)), np.zeros((N,)), 'train_weight')
            ret, net = dec.extract_feature('net.prototxt', 'exp/'+db+'/save_iter_50000.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()
            print("Feature shape", feature.shape)

            gmm_model = dec.TMM(N_class)
            gmm_model.fit(feature)
            # gmm_model.fit(X)
            net.params['loss'][0].data[0,0,:,:] = gmm_model.cluster_centers_.T
            net.params['loss'][1].data[0,0,:,:] = 1.0/gmm_model.covars_.T
        else:
            ret, net = dec.extract_feature('net.prototxt', 'init.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model.cluster_centers_ = net.params['loss'][0].data[0,0,:,:].T


        gmm_model.fit(X)
        Y_pred_last = Y_pred
        Y_pred = gmm_model.predict(X).squeeze()
        # Y_pred = gmm_model.predict(feature).squeeze()
        print(Y_pred)
        # acc, freq = dec.cluster_acc(Y_pred, Y)
        # acc_list.append(acc)
        # nmi = normalized_mutual_info_score(Y, Y_pred)
        # print(freq)
        # print(freq.sum(axis=1))
        # print('acc: ', acc, 'nmi: ', nmi)
        print((Y_pred != Y_pred_last).sum()*1.0/N)
        if (Y_pred != Y_pred_last).sum() < 0.001*N:
            break
        #     print(acc_list)
        #     return (acc, nmi)
        time.sleep(1)

        dec.write_net(db, dim, N_class, "'{:08}'".format(seek))
        weight = gmm_model.transform(X)
        # weight = gmm_model.transform(feature)

        weight = (weight.T/weight.sum(axis=1)).T
        bias = (1.0/weight.sum(axis=0))
        bias = N_class*bias/bias.sum()
        weight = (weight**2)*bias
        weight = (weight.T/weight.sum(axis=1)).T
        print(weight[:10,:])
        dec.write_db(weight, np.zeros((weight.shape[0],)), 'train_weight')

        net.save('init.caffemodel')
        del net

        with open('solver.prototxt', 'w') as fsolver:
            fsolver.write(
"""net: "net.prototxt"
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 100000
display: 10
max_iter: %d
momentum: 0.9
weight_decay: 0.0000
snapshot: 100
snapshot_prefix: "exp/test/save"
snapshot_after_train:true
solver_mode: GPU
debug_info: false
sample_print: false
device_id: 0"""%update_interval
            )
        os.system('caffe train --solver=solver.prototxt --weights=init.caffemodel')
        shutil.copyfile('exp/test/save_iter_%d.caffemodel'%update_interval, 'init.caffemodel')

        iters += 1
        seek = (seek + train_batch_size*update_interval)%N

    for idx, pred in enumerate(Y_pred):
        shutil.copyfile(path.join(input_dir, "%05d.png" % idx), path.join(output_dir, "group%04d" % pred, "%05d.png" % idx))
        print(idx, "->", pred)


def make_image_data(data):
    X = np.asarray(data[:-1]).astype(np.float64) / 255.0
    Y = np.asarray([0] * len(data[:-1]))
    dec.write_db(X, Y, 'image_train')

    X_, Y_ = dec.read_db('image_train', True)
    assert np.abs((X - X_)).mean() < 1e-5
    assert (Y != Y_).sum() == 0

    X2 = np.asarray(data[-1:]).astype(np.float64) / 255.0
    Y2 = np.asarray([0] * len(data[-1:]))
    dec.write_db(X2, Y2, 'image_test')

    X3 = np.concatenate((X,X2), axis=0)
    Y3 = np.concatenate((Y,Y2), axis=0)
    dec.write_db(X3, Y3, 'image_total')


if __name__ == "__main__":
    db = "image"
    input_dir = "../images"
    output_dir = "output"

    img_width = 50
    img_height = 50
    input_dim = img_width * img_height

    data = imgutils.columnize(imgutils.resize_images(imgutils.load_imageset(input_dir, cv2.IMREAD_GRAYSCALE), (img_width, img_height)))

    # make_image_data(data)
    #
    # pretrain.main(db, {
    #     'n_layer': [4],
    #     'dim': [input_dim, 500, 500, 2000, 10],
    #     'drop': [0.0],
    #     'rate': [0.1],
    #     'step': [20000],
    #     'iter': [100000],
    #     'decay': [0.0000]
    # })
    #
    # pretrain.pretrain_main(db, {
    #     'dim': [input_dim, 500, 500, 2000, 10],
    #     'pt_iter': [50000],
    #     'drop': [0.2],
    #     'rate': [0.1],
    #     'step': [20000],
    #     'iter': [100000],
    #     'decay': [0.0000]
    # })
    #
    # os.system("caffe train --solver=ft_solver.prototxt --weights=stack_init_final.caffemodel")
    #
    # DisKmeans()
