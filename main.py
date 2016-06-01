import platform as pl
import numpy as np
from numpy.random.mtrand import RandomState
from scipy import misc
import os
import multiprocessing
import shutil
import tensorflow as tf
import tensorflow.contrib.learn as ln
import sklearn.metrics as me
import sklearn.preprocessing as pp
from sklearn.cross_validation import train_test_split
import random


config = {
    "PrzemoLap": {
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
    "DevBuntu": {
        "labels": "/home/boocheck/datasets/retinopathy/trainLabels.csv",
        "img_full": "/home/boocheck/datasets/retinopathy/full",
        "img_small": "/home/boocheck/datasets/retinopathy/small",
        "img_balanced": "/home/boocheck/datasets/retinopathy/balanced"
    },
    "spark2.opi.org.pl": {
        "labels": "/home/boocheck/datasets/retinopathy/trainLabels.csv",
        "img_full": "/home/boocheck/datasets/retinopathy/full",
        "img_small": "/home/boocheck/datasets/retinopathy/small",
        "img_balanced": "/home/boocheck/datasets/retinopathy/balanced"
    },
    "spark1.opi.org.pl": {
        "labels": "/home/boocheck/datasets/retinopathy/trainLabels.csv",
        "img_full": "/home/boocheck/datasets/retinopathy/full",
        "img_small": "/home/boocheck/datasets/retinopathy/small",
        "img_balanced": "/home/boocheck/datasets/retinopathy/balanced"
    }
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


def read_all_data(dummy_size = 0):
    src_path = conf("img_balanced")

    images = []
    labels = []

    for path, dirs, files in os.walk(src_path):
        for file in files:
            images.append(np.reshape(misc.imread(os.path.join(path, file), False), [-1]))
            labels.append(int(os.path.splitext(file)[0].split("_")[2]))

    res_images = np.stack(images, 0)
    res_labels = np.array(labels)
    res_labels_onehot = np.zeros([len(labels), np.max(res_labels)+1])
    res_labels_onehot[np.arange(len(labels)), res_labels]=1

    if dummy_size > 0:
        dummy_images = []
        dummy_labels = []
        dummy_labels_onehot = []
        for i in xrange(5):
            dummy_indices_i = (res_labels == i).nonzero()[0][:dummy_size]
            print "indices"
            print dummy_indices_i.shape
            print dummy_indices_i

            dummy_images.append(res_images[dummy_indices_i, :])
            print "images"
            print dummy_images[-1].shape
            print dummy_images[-1]

            dummy_labels.extend(res_labels[dummy_indices_i].tolist())
            print "labels"
            print res_labels[dummy_indices_i].shape
            print res_labels[dummy_indices_i]

            dummy_labels_onehot.append(res_labels_onehot[dummy_indices_i, :])
            print "onehot"
            print dummy_labels_onehot[-1].shape
            print dummy_labels_onehot[-1]

        dummy_res_images = np.concatenate(dummy_images, 0)
        dummy_res_labels = np.array(dummy_labels)
        dummy_res_labels_onehot = np.concatenate(dummy_labels_onehot, 0)

        # uniq, indices = np.unique(res_labels, return_index=True)
        # dummy_res_images = res_images[indices, :]
        # dummy_res_labels = res_labels[indices]
        # dummy_res_labels_onehot = res_labels_onehot[indices, :]

        return (res_images.astype(float)/255.0)-0.5, res_labels, res_labels_onehot, (dummy_res_images.astype(float)/255.0)-0.5, dummy_res_labels, dummy_res_labels_onehot

    return (res_images.astype(float)/255.0)-0.5, res_labels, res_labels_onehot

def create_model(inp_h, inp_w, inp_c, conv, fc, dropout=False):
    def model(X, y):

        keep_prob = tf.constant(0.5)

        if len(conv) > 0:
            _conv = [[conv[0][0], conv[0][1], inp_c, conv[0][2]]]
        else:
            _conv = []

        for i in xrange(1, len(conv)):
            _conv.append([conv[i][0], conv[i][1], _conv[i-1][3], conv[i][2]])

        conv_out_h = inp_h
        conv_out_w = inp_w
        conv_out_c = inp_c
        for layer in _conv:
            conv_out_h = conv_out_h - layer[0] + 1
            conv_out_w = conv_out_w - layer[1] + 1
            conv_out_h /= 2
            conv_out_w /= 2
            conv_out_c = layer[3]
        conv_out_neurons = conv_out_h * conv_out_w * conv_out_c

        if len(fc) > 1:
            _fc = [[conv_out_neurons, fc[0]]]
        else:
            _fc = []

        for i in xrange(1, len(fc)):
            _fc.append([_fc[i-1][1], fc[i]])

        conv_weights = [tf.Variable(tf.truncated_normal([x[0], x[1], x[2], x[3]], stddev=0.1)) for x in _conv]
        conv_biases = [tf.Variable(tf.truncated_normal([x[3]], stddev=0.1)) for x in _conv]

        fc_weights = [tf.Variable(tf.truncated_normal([x[0], x[1]], stddev=0.1)) for x in _fc]
        fc_biases = [tf.Variable(tf.truncated_normal([x[1]], stddev=0.1)) for x in _fc]

        h = [tf.reshape(X, [-1, inp_h, inp_w, inp_c])]
        for i in xrange(len(conv_weights)):
            h.append(tf.nn.relu(tf.nn.conv2d(h[-1], conv_weights[i], strides=[1, 1, 1, 1], padding='VALID') + conv_biases[i]))
            h.append(tf.nn.max_pool(h[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))

        h_fc = [tf.reshape(h[-1], [-1, conv_out_neurons])]
        for i in xrange(len(_fc)):
            if i < len(_fc)-1:
                h_fc.append(tf.nn.relu(tf.matmul(h_fc[-1], fc_weights[i]) + fc_biases[i]))
            else:
                h_fc.append(tf.nn.softmax(tf.matmul(h_fc[-1], fc_weights[i]) + fc_biases[i]))

            if dropout and i < len(_fc)-1:
                h_fc.append(ln.ops.dropout(h_fc[-1], keep_prob, name="dropout"+str(i)))

        entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(h_fc[-1], 1e-10, 1.0)))

        return h_fc[-1], entropy

    return model


def random_transform(image, crop_shape, flip=True):
    img_shape = image.shape
    d0 = random.randint(0, img_shape[0]-crop_shape[0])
    d1 = random.randint(0, img_shape[1]-crop_shape[1])
    cropped = image[d0:d0+crop_shape[0], d1:d1+crop_shape[1], :]
    if(flip):
        fliph = random.randint(0,1)
        flipv = random.randint(0,1)
        if(fliph==1):
            cropped = np.fliplr(cropped)
        if(flipv==1):
            cropped = np.flipud(cropped)

    return cropped

def random_data_generator(data, image_shape, crop_shape, flip=True):
    size = data.shape[0]
    rand = random.Random()
    rand.seed(0)
    while True:
        yield random_transform(np.reshape(data[rand.randrange(size), :], image_shape), crop_shape, flip)

def random_labels_generator(labels):
    size = data.shape[0]
    rand = random.Random()
    rand.seed(0)
    while True:
        yield labels[rand.randrange(size)]






if __name__ == '__main__':
    filenames, labels = labels_list(conf("labels"))
    # job_pick_and_move(conf("img_small"), conf("img_balanced"), filenames, labels)
    print "loading dataset..."
    full_data, full_labels, full_onehot, dummy_data, dummy_labels, dummy_onehot = read_all_data(dummy_size=10)
    print "dataset loaded"

    # print "preprocessing dataset..."
    # min_max_scaler = pp.MinMaxScaler((0,1), False)
    # min_max_scaler.fit_transform(data)
    # min_max_scaler.transform(dummy_data)

    # data, labels, onehot = dummy_data, dummy_labels, dummy_onehot
    data, labels, onehot = full_data, full_labels, full_onehot

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.9, random_state=42)

    print data.shape
    print data
    print labels.shape
    print labels
    print onehot.shape
    print onehot

    if os.path.isdir("cls.dump"):
        print "loading estimator..."
        cls = ln.TensorFlowEstimator.restore("cls.dump")
        print "estimator loaded"
    else:
        print "creating model..."
        #
        cls = ln.TensorFlowEstimator(model_fn = create_model(128,128,3, [[3, 3, 16], [4, 4, 32], [5, 5, 64]], [192, 5], True),
                                     n_classes=5, continue_training=True, learning_rate=10e-4, optimizer="RMSProp", steps=100, batch_size=64, config=ln.RunConfig(num_cores=24))
        # cls = ln.dnn.TensorFlowDNNClassifier([3500, 5], n_classes=5, continue_training=True, steps=100, batch_size=128)
        print "model created"
        print "fitting..."
        cls.fit(train_data, train_labels)
        print "fitted"
        print "saving..."
        cls.save("cls.dump")
        print "saved"

    for i in xrange(1000):
            print "i={}".format(i)
            print "fitting..."
            cls.partial_fit(train_data, train_labels)
            print "fitted"
            print "predicting..."
            predicted = cls.predict(test_data)
            print "predicted"
            # print cls.predict_proba(data)
            print "acc={}".format(me.accuracy_score(test_labels, predicted))
            print "confusion matrix"
            print me.confusion_matrix(test_labels, predicted)
            print "saving..."
            cls.save("cls.dump")
            print "saved"




