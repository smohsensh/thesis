import itertools
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.regularizers import l1
from sklearn.cluster.k_means_ import KMeans
from sklearn.neighbors.classification import KNeighborsClassifier
from theano.gof.tests.test_op import utils

import keras.backend as K
from theano import shared
import theano.tensor as T
import numpy as np
import utils


# multi-task network with that predicts attribute on seen data and also
# there is a cost for unseen data relative to their distances from nearest attribute
# hoping this will mitigate domain shift problem
from utils import cluster2vote

activation = 'relu'
optimizer = 'adam'
data_sets = ['apy', 'AwA', 'birds', 'sun']
data = input('Dataset? aPY[1], AwA[2], Birds[3], Sun[4] ')

losses = ['mse', 'mae', 'categorical_crossentropy']
nn_metrics = ['euclidean', 'manhattan']
gammas = [1e-10, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, .001, .05, .01, .1, 1]
epochs = [35]

if data == 1:
    from apy.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
        n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels
    activation = 'sigmoid'
    optimizer = 'rmsprop'
    losses = ['mse', 'mae']
    nn_metrics = ['manhattan']
    gammas = [.01, .001]


if data == 2:
    from awa.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels
    losses = ['categorical_crossentropy']
    nn_metrics = ['euclidean', 'manhattan']
    gammas = [1e-4, 1e-5]
if data == 3:
    from birds.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels
    losses = ['mse', 'mae']
    nn_metrics = ['manhattan']
    gammas = [.01, 1e-5]
if data == 4:
    from sun.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels
    activation = 'sigmoid'
    optimizer = 'adadelta'
    losses = ['categorical_crossentropy', 'mse']
    nn_metrics = ['euclidean']
    gammas = [1e-4, 1e-5]

nb_epoch = 20
loss = 'mse'
n_pretrain = 50
batch_size = 128

test_att = attributes[test_classes].reshape((-1))
test_att = shared(test_att)
n_unseen = nb_unseen_classes

base_net = Sequential()
# base_net.add(Dense(att_dim, activation='sigmoid', input_dim=im_dim, W_regularizer=l1()))
base_net.add(Dense(att_dim, activation=activation, input_dim=im_dim,))
base_net.compile(optimizer, loss)
base_net.fit(tr_feat, attributes[tr_labels], nb_epoch=n_pretrain)

import os

cluster_file = '/home/mohsen/works/data/zsl/comp/clusters/' + data_sets[data] + '.npy'
if os.path.isfile(cluster_file):
    clusters = np.load(cluster_file)
else:
    kmeans = KMeans(len(test_classes), n_jobs=-1)
    clusters = kmeans.fit_predict(test_feat)
    np.save(cluster_file, clusters)

seen_im = Input(shape=(im_dim,))
unseen_im = Input(shape=(im_dim,))
seen_emb = base_net(seen_im)
unseen_emb = base_net(unseen_im)


res_file = open('/home/mohsen/Dropbox/backup/' + data_sets[data-1] + '_min_nearts_res+cluster.csv', 'w')

for params in itertools.product(epochs, losses, nn_metrics, gammas):
    (nb_epoch, loss, nn_metric, gamma) = params

    def domshift_loss(y_true, out):
        dist = K.abs(K.tile(out+y_true, (1, n_unseen)) - test_att)
        dist = dist.reshape((out.shape[0], n_unseen, att_dim))
        dist = K.sum(dist, axis=2)
        mins = K.min(dist, axis=1)
        # distances  = T.sort(dist , axis=1)
        # mins = distances[:,:2]
        return gamma * K.mean(mins)

    g = Model(input=[seen_im, unseen_im], output=[seen_emb, unseen_emb])

    g.compile(optimizer, [loss, domshift_loss])

    n_data = 5*n_train
    tri = np.random.randint(n_train, size=(n_data,))
    tei = np.random.randint(n_test, size=(n_data,))

    g.fit([tr_feat[tri],test_feat[tei]],
            [attributes[tr_labels[tri]], np.zeros_like(attributes[tr_labels[tri]])],
          batch_size=batch_size, nb_epoch=nb_epoch)

    pred = g.predict([test_feat, test_feat])[0]

    nn = KNeighborsClassifier(1, metric=nn_metric)
    nn.fit(attributes[test_classes], test_classes)
    lpred = nn.predict(pred)
    acc = utils.accuracy(lpred, test_labels)
    cluster_ass = cluster2vote(clusters, lpred, n_cluster=len(test_classes))
    cluster_acc  = utils.accuracy(cluster_ass, test_labels)
    s = ('%d, %s, %s,  %f' % params)
    res_file.write(s + (', %f, %f' % (acc, cluster_acc)) + '\n')
    res_file.flush()
    print(s + (', %f, %f' % (acc, cluster_acc)))
