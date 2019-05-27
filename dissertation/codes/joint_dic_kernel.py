import itertools
from sklearn.kernel_ridge import KernelRidge

__author__ = 'mohsen sh'
from scipy.spatial.distance import cdist
import numpy as np

data = input('Dataset? aPY[1], AwA[2], Birds[3], Sun[4] ')
data_sets = ['apy', 'awa', 'birds', 'sun']
if data == 1:
    from apy.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
        n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, instance_level_atts_train
    r0 = np.load('/home/mohsen/works/data/zsl/aPaY/comp/apy_pred.npy')

if data == 2:
    from awa.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels
    r0 = np.load('/home/mohsen/awa_labels.npy')

if data == 3:
    from birds.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, instance_level_atts_train
    r0 = np.load('/home/mohsen/works/data/zsl/birds/bird%52.npy')

if data == 4:
    from sun.data_handle import tr_feat, tr_labels, attributes, att_dim, test_classes, test_labels, test_feat, \
    n_test, n_train, im_dim, nb_unseen_classes, normalized_train_labels, instance_level_atts_train
    pred = np.load('/home/mohsen/works/data/zsl/sun/comp/sun_att.npy')


from utils import accuracy, normalize_class_indexes, to_categorical

###############################
beta = 1
kappa = .001

n_epoch = 80
n_clusters = 36
###############################
def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

r0 = normalize_class_indexes(r0, test_classes)

tr_feat = normalize(tr_feat)
test_feat = normalize(test_feat)
ker = np.mean(cdist(tr_feat, tr_feat))
xs = tr_feat.T
xt = test_feat.T

st = attributes[test_classes].T #a * n_u
ys = attributes[tr_labels].T
if data != 2:
    ys = instance_level_atts_train.T
res_file = open('/home/mohsen/Dropbox/backup/' + data_sets[data-1] + '_dic_joint.csv', 'w')
res_file.write('gamma, lambda, acc \n')

gammas = [5, 15, 50, 1e2, 1e3, 1e4, 2e5]
lambdas = [1, 10, 2e2,1e5, 2e5,]
kernels = np.array([1e-2, 1e-1, 1, 10, 100]) * ker
print ('starting')
for gamma, lambda_, ker in itertools.product(gammas,lambdas, kernels ):

    i = 0
    assignments = r0.copy()
    if max(assignments) !=  nb_unseen_classes - 1:
        assignments[np.random.randint(n_test)] = nb_unseen_classes - 1
    r = to_categorical(assignments) #N_u * n_u
    old_assign = np.zeros_like(assignments)

    while not np.all(assignments == old_assign) and i < n_epoch:
        i += 1
        old_assign = assignments.copy()
        anomaly = np.where(sum(r) < 50)[0]
        for a in anomaly:
            assignments[np.random.randint(n_test, size=(5,))] = a
        r = to_categorical(assignments)

        if np.any(sum(r)==0):
            print('zero in sum (r)' + '-'*30)
        kr = KernelRidge(alpha=gamma, kernel='rbf', gamma=ker)
        yt = r.dot(st.T).T
        all_feat = np.vstack((tr_feat, test_feat))
        all_att = np.vstack((ys.T, yt.T))
        weights = np.ones(n_train+n_test)
        weights[-n_test:] = lambda_
        kr.fit(all_att, all_feat, sample_weight=weights)
        # d = np.dot((xs.dot(ys.T) + lambda_ * xt.dot(r).dot(st.T)), inv)

        # centers = d.dot(st)  # centers: 4096 x #unseen cat
        centers = kr.predict(st.T)
        distances = cdist(xt.T, centers)
        assignments = distances.argmin(axis=-1)
        if max(assignments) != nb_unseen_classes - 1:
            assignments[np.random.randint(n_test)] = nb_unseen_classes - 1
        r = to_categorical(assignments)
    print('finish at %d' % i)
    acc = accuracy(assignments, normalize_class_indexes(test_labels, test_classes))
    print(gamma, lambda_, acc)
    res_file.write('%f, %f, %f \n' % (gamma, lambda_, acc))
    res_file.flush()
res_file.close()
