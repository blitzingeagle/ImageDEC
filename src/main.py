from __future__ import division
from __future__ import print_function

import dec
import pretrain
import imageset_utils as imgutils

import cv2
import numpy as np

import os
import shutil
import time
from glob import glob

import json


def DisKmeans(data, frames_file, target, db="image", update_interval=160):
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

    X = np.asarray(data)
    Y = np.zeros(len(data))
    N = X.shape[0]

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


        Y_pred_last = Y_pred
        Y_pred = gmm_model.predict(feature).squeeze()
        print(Y_pred)

        print((Y_pred != Y_pred_last).sum()*1.0/N)
        if (Y_pred != Y_pred_last).sum() < 0.001*N:
            break
        time.sleep(1)

        dec.write_net(db, dim, N_class, "'{:08}'".format(seek))
        weight = gmm_model.transform(feature)

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

    # Processing files
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    for class_idx in range(N_class):
        group_dir = os.path.join(output_dir, "group%04d" % class_idx)
        os.makedirs(group_dir)

    if os.path.isfile(frames_file):
        with open(frames_file) as in_file:
            json_lines = in_file.readlines()
            json_lines = [json.loads(x) for x in json_lines]
    else:
        json_lines = [{}] * N

    file_list = sorted(glob(os.path.join(input_dir, "*")))
    for idx, pred in enumerate(Y_pred):
        filename = os.path.basename(file_list[idx])
        shutil.copyfile(file_list[idx], os.path.join(output_dir, "group%04d" % pred, filename))

        name_segs = os.path.splitext(filename)[0].split("_")
        frame_num = int(name_segs[1])
        obj_num = int(name_segs[3])

        tag_item = json_lines[frame_num - 1]["tag"][obj_num - 1]
        tag_item["cluster"] = chr(ord('A') + pred)
        tag_item["object_filename"] = filename

        print(filename, "->", pred)

    with open(frames_file.replace("frame.txt", "cluster.txt"), "w") as out_file:
        # Filter content
        tag_keys = ["top", "bot", "left", "right", "cluster", "object_filename", target]
        for line in json_lines:
            line["tag"] = [{key:tag[key] for key in tag_keys if key in tag} for (idx, tag) in enumerate(line["tag"]) if target in tag.keys()]

            if len(line["tag"]) > 0:
                out_file.write(json.dumps(line, sort_keys=True))
                out_file.write("\n")



def make_data(data, db="image"):
    db_path = os.path.join("modules", db, "database")
    os.system("mkdir -p " + db_path)

    db_train = os.path.join(db_path, "train")
    db_test = os.path.join(db_path, "test")
    db_total = os.path.join(db_path, "total")

    X = np.asarray(data[:-1]).astype(np.float64) / 255.0
    Y = np.asarray([0] * len(data[:-1]))
    dec.write_db(X, Y, db_train)

    X_, Y_ = dec.read_db(db_train, True)
    assert np.abs((X - X_)).mean() < 1e-5
    assert (Y != Y_).sum() == 0

    X2 = np.asarray(data[-1:]).astype(np.float64) / 255.0
    Y2 = np.asarray([0] * len(data[-1:]))
    dec.write_db(X2, Y2, db_test)

    X3 = np.concatenate((X,X2), axis=0)
    Y3 = np.concatenate((Y,Y2), axis=0)
    dec.write_db(X3, Y3, db_total)


if __name__ == "__main__":
    db = "image"

    data_path = "../output_00/15/15_Evening_1/"
    target = "car"
    input_dir = os.path.join(data_path, target)
    frames_file = os.path.join(data_path, "frame.txt")
    output_dir = "output"
    option = None

    img_width = 50
    img_height = 50

    if option == None:
        img_channels = 4
        input_dim = img_width * img_height * img_channels
        data1 = imgutils.columnize(imgutils.resize_images(imgutils.load_imageset(input_dir, cv2.IMREAD_COLOR), (img_width, img_height)))
        data2 = imgutils.columnize(imgutils.resize_images(imgutils.load_imageset(input_dir, cv2.IMREAD_GRAYSCALE), (img_width, img_height)))
        data = np.hstack([data1,data2])
        print(data.shape)
    else:
        img_channels = 3 if option == cv2.IMREAD_COLOR else 1
        input_dim = img_width * img_height * img_channels
        data = imgutils.columnize(imgutils.resize_images(imgutils.load_imageset(input_dir, option), (img_width, img_height)))

    mod_path = os.path.join("modules", db)
    os.system("mkdir -p " + mod_path)

    if not os.path.exists(mod_path):
        make_data(data, db)

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
        os.system("mv -t {} pt_net.prototxt ft_solver.prototxt stack_net.prototxt pt_solver.prototxt stack_init.caffemodel stack_init_final.caffemodel".format(mod_path))

    DisKmeans(data, frames_file, target, db=db)
