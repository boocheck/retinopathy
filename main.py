import platform as pl
import numpy as np
from numpy.random.mtrand import RandomState
from scipy import misc
import os
import multiprocessing
import shutil
import tensorflow as tf
import tensorflow.contrib.learn as ln


config = {
    "home_host": {
        "labels": "D:\\Datasets\\retinopathy\\trainLabels.csv",
        "img_full": "D:\\Datasets\\retinopathy\\trainLabels.csv",
        "img_small": "D:\\Datasets\\retinopathy\\trainLabels.csv",
        "img_balanced": "D:\\Datasets\\retinopathy\\trainLabels.csv"

    },
    "LIS-BUCZKOWSKI": {
        "labels": "D:\\datasets\\retinopathy\\trainLabels.csv",
        "img_full": "D:\\datasets\\retinopathy\\full",
        "img_small": "D:\\datasets\\retinopathy\\small",
        "img_balanced": "D:\\datasets\\retinopathy\\balanced"
    },
}


def conf(key, platform=None):
    if platform == None:
        platform = pl.node()
    return config[platform][key]

def n_sized_chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def n_chunks(l, n):
    size = len(l)/n
    rest = len(l)%n

    for i in xrange(n):
        yield l[i*size + min(i, rest): (i+1)*size + min(i+1, rest)]


def labels_list(path):
    list = np.loadtxt(path, str, skiprows=1, delimiter=",").tolist()
    unzipped = zip(*list)
    return np.array(unzipped[0]), np.array(map(lambda x: int(x), unzipped[1]))


def resize_single_image(full_path_src, full_path_dst, shape_dst):
    if not os.path.exists(full_path_dst):
        img = misc.imread(full_path_src)
        shape = img.shape
        shorter = min(shape[0], shape[1])
        start_0, start_1 = (shape[0]-shorter)/2, (shape[1]-shorter)/2
        img = img[start_0:start_0+shorter, start_1:start_1+shorter, :]
        img = misc.imresize(img, shape_dst, interp="bicubic")
        misc.imsave(full_path_dst, img)


def curried_resize(arg):
        resize_single_image(arg[0], arg[1], arg[2])
        return None


def job_resize(path_src, path_dst, shape_dst, processess = 1):
    full_path_list_src = []
    for path, dirs, files in os.walk(path_src):
        for file in files:
            full_path_list_src.append(os.path.join(path, file))

    full_path_list_dst = [os.path.join(path_dst, os.path.splitext(os.path.basename(p))[0] + ".png") for p in full_path_list_src]
    paths = zip(full_path_list_src, full_path_list_dst, [shape_dst] * len(full_path_list_src))

    pool = multiprocessing.Pool(processess)
    pool.map(curried_resize, paths)


def job_pick_and_move(path_src, path_dst, filenames, labels):
    rs = RandomState(0)
    bools = [labels == l for l in xrange(5)]
    indices_lists = [x.nonzero()[0] for x in bools]
    permuted_indices = [rs.permutation(x) for x in indices_lists]
    permuted_indices_top700 = [x[:700] for x in permuted_indices]

    for i in xrange(5):
        for s in filenames[permuted_indices_top700[i]]:
            shutil.copy(os.path.join(path_src, s+".png"), os.path.join(path_dst, s + "_" + str(i) + ".png"))


def read_all_data():
    src_path = conf("img_balanced")

    images = []
    labels = []

    for path, dirs, files in os.walk(src_path):
        for file in files:
            images.append(misc.imread(os.path.join(path, file), True))
            labels.append(int(os.path.splitext(file)[0].split("_")[2]))

    return np.concatenate(images, 1), np.array(labels)



def create_model(inp_h, inp_w, inp_c, conv, fc):
    def model(X, y, dropout):

        keep_prob = tf.placeholder("float", [], "keep_prob")

        _conv = [[conv[0][0], conv[0][1], conv[0][inp_c], conv[0][4]]]
        for i in xrange(1, len(conv)):
            _conv.append([conv[i][0], conv[i][1], _conv[i-1][4], conv[i][4]])

        conv_out_h = inp_h
        conv_out_w = inp_w
        for layer in _conv:
            conv_out_h = conv_out_h - layer[0] + 1
            conv_out_w = conv_out_w - layer[0] + 1
            conv_out_h /= 2
            conv_out_w /= 2
        conv_out_neurons = conv_out_h * conv_out_w * conv[-1][3]

        _fc = [[conv_out_neurons, fc[0]]]
        for i in xrange(1, len(fc)):
            _fc.append([_fc[i-1], fc[i]])

        conv_weights = [tf.Variable(tf.truncated_normal([x[0], x[1], x[2], x[3]], stddev=0.1)) for x in _conv]
        conv_biases = [tf.Variable(tf.truncated_normal([x[3]], stddev=0.1)) for x in _conv]

        fc_weights = [tf.Variable(tf.truncated_normal([x[0], x[1]], stddev=0.1)) for x in _fc]
        fc_biases = [tf.Variable(tf.truncated_normal([x[1]], stddev=0.1)) for x in _fc]


        h = [ tf.reshape(X, [-1, inp_h, inp_w, inp_c])]
        for i in xrange(len(conv_weights)):
            h.append(tf.tanh(tf.nn.conv2d(h[-1], conv_weights[i], strides=[1, 1, 1, 1], padding='VALID') + conv_biases[i]))

        h_fc = [tf.reshape(h[-1], [-1, conv_out_neurons])]
        for i in xrange(len(_fc)):
            h_fc.append(tf.tanh(tf.matmul(h_fc[-1], fc_weights[i]) + fc_biases[i]))
            if dropout:
                h_fc.append(ln.ops.dropout(h_fc[-1], 0.5))

        softmax = tf.nn.softmax(h_fc[-1])
        entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)))

        # tmpVars = tf.Variable(tf.truncated_normal([276, 2], stddev=0.1))
        # tmpBias = tf.Variable(tf.truncated_normal([2], stddev=0.1))
        # tmpOutput = tf.tanh(tf.matmul(X, tmpVars) + tmpBias)
        # tmpErr = tf.reduce_mean(tf.square(y-tmpOutput))

        return softmax, entropy
        # return tmpOutput, tmpErr
    return aaia_model


if __name__ == '__main__':
    filenames, labels = labels_list(conf("labels"))
    # job_pick_and_move(conf("img_small"), conf("img_balanced"), filenames, labels)
    data, labels = read_all_data()
    x= 5




