import os
os.environ['PATH'] = '../caffe/build/tools:' + os.environ['PATH']
import sys
sys.path = ['../caffe/python'] + sys.path

import dec
import numpy as np
import caffe
import scipy.io
import shutil


# Creates pt_net.prototxt and ft_solver.prototxt
def define_solver(db, params):
    n_layer = params['n_layer'][0]
    drop = params['drop'][0]

    mod_path = os.path.join("modules", db)
    db_path = os.path.join(mod_path, "database")
    exp_path = os.path.join(mod_path, "exp")
    save_prefix = exp_path + "/save"

    encoder_layers = [ ('data', ('data','label', os.path.join(db_path, "train"), os.path.join(db_path, "test"), 1.0)) ]
    decoder_layers = [ ('euclid', ('pt_loss', 'd_data', 'data')) ]
    last_dim = params['dim'][0]
    niter = params['iter'][0]
    rate = params['rate'][0]
    for i in xrange(n_layer):
        str_h1 = 'inner%d'%(i+1)
        str_h2 = 'd_inner%d'%(i+1)
        str_x = 'inner%d'%(i)
        str_y = 'd_inner%d'%(i)
        dim = params['dim'][i+1]
        if i == 0:
            str_x = 'data'
            str_y = 'd_data'
        if i == n_layer-1:
            str_h1 = 'output'
            str_h2 = 'output'
        if i != n_layer-1:
            encoder_layers.extend([
                ('inner_init', (str_h1, str_x, dim, np.sqrt(1.0/last_dim))),
                ('relu', (str_h1,)),
                ('drop', (str_h1, drop)),
                ])
        else:
            encoder_layers.extend([
                ('inner_init', (str_h1, str_x, dim, np.sqrt(1.0/last_dim))),
                ('drop', (str_h1, drop)),
                ])
        if i != 0:
            decoder_layers.append(('drop', (str_y, drop)))
            decoder_layers.extend([
                ('relu', (str_y,)),
                ('inner_init', (str_y, str_h2, last_dim, np.sqrt(1.0/dim)))
                ])
        else:
            decoder_layers.extend([
                ('inner_init', (str_y, str_h2, last_dim, np.sqrt(1.0/dim)))

                ])
        last_dim = dim
    with open(os.path.join(mod_path, "pt_net.prototxt"), 'w') as fnet:
        dec.make_net(fnet, encoder_layers+decoder_layers[::-1])

    with open(os.path.join(mod_path, "ft_solver.prototxt"), 'w') as fsolver:
        template = ""
        with open("templates/ft_solver_template.txt", "r") as template_file:
            template = template_file.read()
        fsolver.write(template.format(mod_path, rate, params['step'][0], niter, params['decay'][0], save_prefix))


def initialize_model(db, params):
    dim = params['dim']
    n_layer = len(dim)-1

    w_down = []
    b_down = []

    mod_path = os.path.join("modules", db)
    db_path = os.path.join(mod_path, "database")
    exp_path = os.path.join(mod_path, "exp")
    save_prefix = exp_path + "/save"

    for i in xrange(n_layer):
        rate = params['rate'][0]
        layers = [ ('data', ('data','label', os.path.join(db_path, "train"), os.path.join(db_path, "test"), 1.0)) ]
        str_x = 'data'
        for j in xrange(i):
            str_h = 'inner%d'%(j+1)
            layers.extend([
                    ('inner_lr', (str_h, str_x, dim[j+1], 0.05, 0.0, 0.0)),
                    ('relu', (str_h,)),
                ])
            str_x = str_h
        if i == n_layer-1:
            str_h = 'output'
        else:
            str_h = 'inner%d'%(i+1)
        if i != 0:
            layers.extend([
                        ('drop_copy', (str_x+'_drop', str_x, params['drop'][0])),
                        ('inner_init', (str_h, str_x+'_drop', dim[i+1], 0.01)),
                ])
        else:
            layers.extend([
                        ('drop_copy', (str_x+'_drop', str_x, 0.0)),
                        ('inner_init', (str_h, str_x+'_drop', dim[i+1], 0.01)),
                ])
        if i != n_layer-1:
            layers.append(('relu', (str_h,)))
            layers.append(('drop', (str_h, params['drop'][0])))
        layers.append(('inner_init', ('d_'+str_x, str_h, dim[i], 0.01)))
        if i != 0:
            layers.append(('relu', ('d_'+str_x,)))
            layers.append(('euclid', ('pt_loss%d'%(i+1), 'd_'+str_x, str_x)))
        else:
            layers.append(('euclid', ('pt_loss%d'%(i+1), 'd_'+str_x, str_x)))

        with open(os.path.join(mod_path, "stack_net.prototxt"), 'w') as fnet:
            dec.make_net(fnet, layers)

        with open(os.path.join(mod_path, "pt_solver.prototxt"), 'w') as fsolver:
            template = ""
            with open("templates/pt_solver_template.txt", 'r') as template_file:
                template = template_file.read()
            fsolver.write(template.format(mod_path, rate, params['step'][0], params['pt_iter'][0], params['decay'][0], save_prefix))

        if i > 0:
            model = save_prefix + '_iter_%d.caffemodel'%params['pt_iter'][0]
        else:
            model = None

        mean, net = dec.extract_feature(os.path.join(mod_path, "stack_net.prototxt"), model,
                                        [str_x], 1, train=True, device=0)

        net.save(os.path.join(mod_path, "stack_init.caffemodel"))

        os.system("mkdir -p " + exp_path)
        if not os.system("caffe train --solver={0} --weights={1}".format(os.path.join(mod_path, "pt_solver.prototxt"), os.path.join(mod_path, "stack_init.caffemodel"))) == 0:
            print("Caffe failed")

        net = caffe.Net(os.path.join(mod_path, "stack_net.prototxt"), save_prefix + '_iter_%d.caffemodel'%params['pt_iter'][0])
        w_down.append(net.params['d_'+str_x][0].data.copy())
        b_down.append(net.params['d_'+str_x][1].data.copy())
        del net

    net = caffe.Net(os.path.join(mod_path, "pt_net.prototxt"), save_prefix + '_iter_%d.caffemodel'%params['pt_iter'][0])
    for i in xrange(n_layer):
        if i == 0:
            k = 'd_data'
        else:
            k = 'd_inner%d'%i
        net.params[k][0].data[...] = w_down[i]
        net.params[k][1].data[...] = b_down[i]
    net.save(os.path.join(mod_path, "stack_init_final.caffemodel"))


def export_model(db, iters, dest="ft_export.caffemodel"):
    mod_path = os.path.join("modules", db)
    save_prefix = os.path.join(mod_path, "exp") + "/save"

    os.system("caffe train --solver={0} --weights={1}".format(os.path.join(mod_path, "ft_solver.prototxt"), os.path.join(mod_path, "stack_init_final.caffemodel")))

    shutil.copyfile(save_prefix + '_iter_%d.caffemodel'%iters, dest)


if __name__ == '__main__':
    db = "mnist5"       # Database name
    input_dim = 784     # Dimension for input images (28x28=784)
    iters = 100000

    mod_path = os.path.join("modules", db)
    os.system("mkdir -p " + mod_path)

    if not os.path.exists(os.path.join(mod_path, "database")):
        dec.make_mnist_data(db)   # Prepare database for initial training

    if not os.path.exists(os.path.join(mod_path, "pt_net.prototxt")) or not os.path.exists(os.path.join(mod_path, "ft_solver.prototxt")):
        define_solver(db, {
            'n_layer': [4],
            'dim': [input_dim, 500, 500, 2000, 10],
            'drop': [0.0],
            'rate': [0.1],
            'step': [20000],
            'iter': [iters],
            'decay': [0.0000]
        })

    if not os.path.exists(os.path.join(mod_path, "stack_init_final.caffemodel")):
        initialize_model(db, {
            'dim': [input_dim, 500, 500, 2000, 10],
            'pt_iter': [50000],
            'drop': [0.2],
            'rate': [0.1],
            'step': [20000],
            'iter': [iters],
            'decay': [0.0000]
        })

    if not os.path.exists(os.path.join(mod_path, "ft_export.caffemodel")):
        export_model(db, iters)
