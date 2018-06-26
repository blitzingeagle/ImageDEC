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

from argparse import ArgumentParser


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


def DisKmeans(data, target, db="image", dim=10, N_class = 5, update_interval=100):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    from sklearn.lda import LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.spatial.distance import cdist
    import cPickle
    from scipy.io import loadmat

    mod_path = os.path.join("modules", db)
    net_proto = os.path.join(mod_path, "net.prototxt")
    init_model = os.path.join(mod_path, "init.caffemodel")
    train_weight = os.path.join(mod_path, "train_weight")
    solver_proto = os.path.join(mod_path, "solver.prototxt")
    test_prefix = os.path.join(mod_path, "exp") + "/test"

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

    acc_list = []

    while True:
        dec.write_net(db, dim, N_class, "'{:08}'".format(0))

        if iters == 0:
            dec.write_db(np.zeros((N,N_class)), np.zeros((N,)), train_weight)
            ret, net = dec.extract_feature(net_proto, os.path.join(mod_path, "ft_export.caffemodel"), ['output'], N, True, 0)
            feature = ret[0].squeeze()

            gmm_model = dec.TMM(N_class)
            gmm_model.fit(feature)
            net.params['loss'][0].data[0,0,:,:] = gmm_model.cluster_centers_.T
            net.params['loss'][1].data[0,0,:,:] = 1.0/gmm_model.covars_.T
        else:
            ret, net = dec.extract_feature(net_proto, init_model, ['output'], N, True, 0)
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
        transform = weight

        weight = (weight.T/weight.sum(axis=1)).T
        bias = (1.0/weight.sum(axis=0))
        bias = N_class*bias/bias.sum()
        weight = (weight**2)*bias
        weight = (weight.T/weight.sum(axis=1)).T
        print(weight[:10,:])
        dec.write_db(weight, np.zeros((weight.shape[0],)), train_weight)

        net.save(init_model)
        del net

        with open(solver_proto, 'w') as fsolver:
            template = ""
            with open("templates/solver_template.txt", "r") as template_file:
                template = template_file.read()
            fsolver.write(template.format(net_proto, update_interval, test_prefix))

        os.system('caffe train --solver={0} --weights={1}'.format(solver_proto, init_model))
        shutil.copyfile(test_prefix + '_iter_%d.caffemodel'%update_interval, init_model)

        iters += 1
        seek = (seek + train_batch_size*update_interval)%N

    cluster_centers = gmm_model.cluster_centers_.tolist()
    transform = transform.tolist()

    return (Y_pred, cluster_centers, transform)


parser = ArgumentParser("Produce clusters from image data.")
parser.add_argument("-db", "--database", type=str, default="image30x30_dim50", metavar="DATABASE", help="Database name for cluster model.")
parser.add_argument("-c", "--classes", type=int, default=5, metavar="CLASSES", help="Number of classes of clustering.")
parser.add_argument("-dim", "--dimensions", type=int, default=50, metavar="DIMENSIONS", help="Dimensionality of cluster data.")
parser.add_argument("-iw", "--width", type=int, default=30, metavar="WIDTH", help="Width of cluster data.")
parser.add_argument("-ih", "--height", type=int, default=30, metavar="HEIGHT", help="Height of cluster data.")
parser.add_argument("-t", "--target", type=str, metavar="TARGET", help="Target name.")
parser.add_argument("-p", "--path", type=str, metavar="PATH", help="Path of frames.")
parser.add_argument("-loc", "--location", type=str, default=".", metavar="LOC", help="Caller location.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.target is None or args.path is None:
        print("Needs target and path.")
        parser.print_help()
        exit(0)

    # Settings
    db = args.database
    iters = 100000
    dim = args.dimensions
    N_class = args.classes
    img_width = args.width
    img_height = args.height

    data_path = os.path.join(args.location, args.path)
    target = args.target
    frames_file = os.path.join(data_path, "frame.txt")

    input_dir = os.path.join(data_path, target)
    output_dir = os.path.join(input_dir, "clusters")
    option = None

    image_paths = imgutils.image_paths(input_dir)

    if option == None:
        img_channels = 4
        input_dim = img_width * img_height * img_channels
        data1 = imgutils.columnize(imgutils.resize_images(imgutils.load_images(image_paths, cv2.IMREAD_COLOR), (img_width, img_height)))
        data2 = imgutils.columnize(imgutils.resize_images(imgutils.load_images(image_paths, cv2.IMREAD_GRAYSCALE), (img_width, img_height)))
        data = np.hstack([data1,data2])
        print(data.shape)
    else:
        img_channels = 3 if option == cv2.IMREAD_COLOR else 1
        input_dim = img_width * img_height * img_channels
        data = imgutils.columnize(imgutils.resize_images(imgutils.load_images(image_paths, option), (img_width, img_height)))


    mod_path = os.path.join("modules", db)
    os.system("mkdir -p " + mod_path)


    # Pretrain
    if not os.path.exists(os.path.join(mod_path, "database")):
        make_data(data, db)

    if not os.path.exists(os.path.join(mod_path, "pt_net.prototxt")) or not os.path.exists(os.path.join(mod_path, "ft_solver.prototxt")):
        pretrain.define_solver(db, {
            'n_layer': [4],
            'dim': [input_dim, 500, 500, 2000, dim],
            'drop': [0.0],
            'rate': [0.1],
            'step': [20000],
            'iter': [iters],
            'decay': [0.0000]
        })

    if not os.path.exists(os.path.join(mod_path, "stack_init_final.caffemodel")):
        pretrain.initialize_model(db, {
            'dim': [input_dim, 500, 500, 2000, dim],
            'pt_iter': [50000],
            'drop': [0.2],
            'rate': [0.1],
            'step': [20000],
            'iter': [iters],
            'decay': [0.0000]
        })

    if not os.path.exists(os.path.join(mod_path, "ft_export.caffemodel")):
        pretrain.export_model(db, iters)


    (Y_pred, cluster_centers, transform) = DisKmeans(data, target, db=db, dim=dim, N_class=N_class)

    # Processing files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    for class_idx in range(N_class):
        group_dir = os.path.join(output_dir, "group%04d" % class_idx)
        os.makedirs(group_dir)

    with open(os.path.join(output_dir, "cluster_centers.txt"), "w") as file:
        matrix = cluster_centers
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        file.write('\n'.join(table))
    with open(os.path.join(output_dir, "transform_weight.txt"), "w") as file:
        matrix = transform
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        file.write('\n'.join(table))


    if os.path.isfile(frames_file):
        with open(frames_file) as in_file:
            json_lines = in_file.readlines()
            json_lines = [json.loads(x) for x in json_lines]
    else:
        json_lines = [{}] * N


    weighted_data = [[] for i in range(N_class)]

    for idx, pred in enumerate(Y_pred):
        # Copy file to group directory
        filename = os.path.basename(image_paths[idx])
        shutil.copyfile(image_paths[idx], os.path.join(output_dir, "group%04d" % pred, filename))

        # Write to weighted_data
        weighted_data[pred].append((filename, transform[idx][pred]))

        # Write cluster data to JSON
        name_segs = os.path.splitext(filename)[0].split("_")
        frame_num = int(name_segs[1])
        obj_num = int(name_segs[3])

        tag_item = json_lines[frame_num - 1]["tag"][obj_num - 1]
        tag_item["cluster"] = chr(ord('A') + pred)
        tag_item["object_filename"] = os.path.join(args.path, filename)

        print(filename, "->", pred, end=("\n" if idx % 4 == 3 else "\t\t"))

    print()

    group_files = [open(os.path.join(output_dir, "group%04d" % x, "cluster_info.txt"), 'w') for x in range(N_class)]

    for (idx, weighted_group) in enumerate(weighted_data):
        weighted_group = sorted(weighted_group, reverse=False)
        for (filename, weight) in weighted_group:
            group_files[idx].write("{} {}\n".format(filename, weight))

    for file in group_files:
        file.close()

    # print("Cluster Centers:", gmm_model.cluster_centers_)
    # print("Transform:", transform.shape)


    with open(frames_file.replace("frame.txt", "cluster.txt"), "w") as out_file:
        # Filter content
        tag_keys = ["top", "bot", "left", "right", "cluster", "object_filename", target]
        for line in json_lines:
            line["tag"] = [{key:tag[key] for key in tag_keys if key in tag} for (idx, tag) in enumerate(line["tag"]) if target in tag.keys()]

            if len(line["tag"]) > 0:
                out_file.write(json.dumps(line, sort_keys=True))
                out_file.write("\n")
