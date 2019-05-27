from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors.classification import KNeighborsClassifier
from clusterings import kmeans_modifed_assingment

__author__ = 'mohsen'
from utils import normalize_class_indexes, accuracy, to_categorical
import itertools
import numpy as np



def load_data(data):
    global r0, pred
    if data == 1:
        from apy.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
            n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, nb_seen_classes, train_classes
        r0 = np.load('/home/mohsen/works/data/zsl/aPaY/comp/apy_pred.npy')
    if data == 2:
        from awa.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
            n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, nb_seen_classes, train_classes
        r0 = np.load('/home/mohsen/awa72%labels.npy')
    if data == 3:
        from birds.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
            n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, nb_seen_classes, train_classes
        r0 = np.load('/home/mohsen/works/data/zsl/birds/bird%52.npy')
    if data == 4:
        from sun.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
            n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, nb_seen_classes, train_classes
        r0 = np.load('/home/mohsen/works/data/zsl/sun/comp/pred%85.npy')

    return attributes, n_test, nb_seen_classes, nb_unseen_classes, test_classes, test_feat, test_labels, tr_feat, tr_labels, train_classes


data_sets = ['apy', 'awa', 'birds', 'sun']
data = input('Dataset? aPY[1], AwA[2], Birds[3], Sun[4] ')

attributes, n_test, nb_seen_classes, nb_unseen_classes, test_classes, test_feat, test_labels, tr_feat, tr_labels, train_classes = load_data(data)

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)



# rcv = RidgeCV(alphas=[.001, .1, 1, 5, 15, 50, 1e2, 2e2, 5e2, 1e3], cv=20)
# rcv.fit(xs.T, ys.T)
res_file = open('/home/mohsen/Dropbox/backup/kmeans_' + data_sets[data - 1] + '.csv', 'a')
res_file.write('gamma, nclusteer, beta,  acc \n')

gammas = [.001, .01, .1, 1, 5, 100, 500, 1e3, 1e4, 1e5, 1e6]


import os

cluster_file = '/home/mohsen/works/data/zsl/comp/clusters/2' + data_sets[data -1] + '.npy'
center_file = '/home/mohsen/works/data/zsl/comp/clusters/2centers_' + data_sets[data -1] + '.npy'
if os.path.isfile(cluster_file) and os.path.isfile(center_file):
    clusters = np.load(cluster_file)
    centers = np.load(center_file)
else:
    n_clusters = nb_seen_classes + 2 * nb_unseen_classes
    clusters, centers = kmeans_modifed_assingment(np.vstack((tr_feat, test_feat)),
                                                  n_clusters, y=normalize_class_indexes(tr_labels, train_classes), )
    clusters = clusters[-n_test:]
    np.save(cluster_file, clusters)
    np.save(center_file, centers)


tr_feat = normalize(tr_feat)
test_feat = normalize(test_feat)

for gamma in gammas:
    att = attributes[tr_labels]
    inv = np.linalg.inv(np.dot(att.T, att) + gamma * np.eye(att.shape[1]))
    ds = np.dot(inv, att.T.dot(tr_feat))
    landmarks = np.dot(attributes[test_classes], ds)
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(landmarks, test_classes)
    cl_pred = nn.predict(centers)
    pred = cl_pred[clusters]
    acc = accuracy(pred, test_labels)
    print(gamma, acc)

    res_file.write('%f, %f \n' % (gamma, acc))
    res_file.flush()
res_file.close()
