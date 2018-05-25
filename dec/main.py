from __future__ import division
from __future__ import print_function

import dec

import numpy as np
import cv2

import os.path as path
from glob import glob
from operator import mul

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

    if db == 'mnist':
        N_class = 10
        batch_size = 100
        train_batch_size = 256
        X, Y = read_db(db+'_total', True)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(np.squeeze(Y), dtype = np.int32)
        N = X.shape[0]
        img = np.clip((X/0.02), 0, 255).astype(np.uint8).reshape((N, 28, 28, 1))

    tmm_alpha = 1.0
    total_iters = (N-1)/train_batch_size+1
    if not update_interval:
      update_interval = total_iters
    Y_pred = np.zeros((Y.shape[0]))
    iters = 0
    seek = 0
    dim = 10


    acc_list = []

    while True:
        write_net(db, dim, N_class, "'{:08}'".format(0))
        if iters == 0:
            write_db(np.zeros((N,N_class)), np.zeros((N,)), 'train_weight')
            ret, net = extract_feature('net.prototxt', 'exp/'+db+'/save_iter_100000.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model = TMM(N_class)
            gmm_model.fit(feature)
            net.params['loss'][0].data[0,0,:,:] = gmm_model.cluster_centers_.T
            net.params['loss'][1].data[0,0,:,:] = 1.0/gmm_model.covars_.T
        else:
            ret, net = extract_feature('net.prototxt', 'init.caffemodel', ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model.cluster_centers_ = net.params['loss'][0].data[0,0,:,:].T


        Y_pred_last = Y_pred
        Y_pred = gmm_model.predict(feature).squeeze()
        acc, freq = cluster_acc(Y_pred, Y)
        acc_list.append(acc)
        nmi = normalized_mutual_info_score(Y, Y_pred)
        print(freq)
        print(freq.sum(axis=1))
        print('acc: ', acc, 'nmi: ', nmi)
        print (Y_pred != Y_pred_last).sum()*1.0/N
        if (Y_pred != Y_pred_last).sum() < 0.001*N:
            print(acc_list)
        return acc, nmi
        time.sleep(1)

        write_net(db, dim, N_class, "'{:08}'".format(seek))
        weight = gmm_model.transform(feature)

        weight = (weight.T/weight.sum(axis=1)).T
        bias = (1.0/weight.sum(axis=0))
        bias = N_class*bias/bias.sum()
        weight = (weight**2)*bias
        weight = (weight.T/weight.sum(axis=1)).T
        print weight[:10,:]
        write_db(weight, np.zeros((weight.shape[0],)), 'train_weight')

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


if __name__ == "__main__":
    input_dir = "../images"
    imageset = resize_to_mean(load_imageset("images", cv2.IMREAD_GRAYSCALE))
    data = columnize(imageset)

    DisKmeans()
