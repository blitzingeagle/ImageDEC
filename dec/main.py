from __future__ import division
from __future__ import print_function

import dec
import pretrain

import numpy as np
import cv2

import os
import os.path as path
from glob import glob
from operator import mul
import shutil
import time

# === Imageset Loading ===
def image_paths(directory):
    return sorted(glob(path.join(directory, '*')))


def load_images(filepaths, option=cv2.IMREAD_COLOR):
    return [cv2.imread(filepath, option) for filepath in filepaths]


def load_imageset(directory, option=cv2.IMREAD_COLOR):
    return load_images(image_paths(directory), option)


# === Imageset Stats ===
def imageset_stats(imageset):
    stats = {}
    count = len(imageset)
    shape = {}

    height_sum, width_sum = 0, 0
    for img in imageset:
        height_sum += img.shape[0]
        width_sum += img.shape[1]
    shape["mean_height"] = height_sum / count
    shape["mean_width"] = width_sum / count

    stats["count"] = count
    stats["shape"] = shape

    return stats


# === Resize Images ===
def resize_images(images, shape=(128,128)):
    return [cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC) for img in images]


def resize_to_mean(imageset):
    imgset_stats = imageset_stats(imageset)
    target_shape = (int(imgset_stats["shape"]["mean_width"]), int(imgset_stats["shape"]["mean_height"]))
    return resize_images(imageset, shape=target_shape)


# === Columnize ===
def columnize(dataset):
    return [elem.reshape(reduce(mul, elem.shape, 1)) for elem in dataset]


def DisKmeans():
    input_dir = "images"
    imageset = resize_images(load_imageset(input_dir, cv2.IMREAD_GRAYSCALE), (50, 50))
    data = columnize(imageset)
    print("Data Loaded")

    db = "mnist"
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
            ret, net = dec.extract_feature('net.prototxt', 'exp/'+db+'/save_iter_100000.caffemodel', ['output'], N, True, 0)
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

if __name__ == "__main__":
    db = "mnist"
    input_dim = 784
    dec.make_mnist_data()

    pretrain.main(db, {
        'n_layer': [4],
        'dim': [input_dim, 500, 500, 2000, 10],
        'drop': [0.0],
        'rate': [0.1],
        'step': [20000],
        'iter': [100000],
        'decay': [0.0000]
    })

    pretrain.pretrain_main(db, {
        'dim': [input_dim, 500, 500, 2000, 10],
        'pt_iter': [50000],
        'drop': [0.2],
        'rate': [0.1],
        'step': [20000],
        'iter': [100000],
        'decay': [0.0000]
    })

    os.system("caffe train --solver=ft_solver.prototxt --weights=stack_init_final.caffemodel")

    # DisKmeans()
