from operator import mul
import os
import random
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import cdist


def string2list(s, dtype='float'):
    if dtype == 'float':
        return [float(ord(x)) for x in s]     
    elif dtype == 'int':
        return [int(ord(x)) for x in s]
    else:
        return None

def tuples_prod(l):
    return reduce(mul,[x1*x2 for (x1, x2) in l])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def get_rand_except(min, max, ex):
    r = random.randint(min, max)
    while r == ex:
        r = random.randint(min, max)
    return r


def zero_pad(num, target_length=4):
    num = str(num)
    return '0'*(target_length - len(num)) + num



def normalize_class_indexes(indices, distinct_labels):
    """
    normalize class labels to contain only 0 to #classes -1
    :param indices: y_train or y_test, etc
    :param distinct_labels: list of labels used in indices
    :return:
    """
    d = {}
    for i, l in enumerate(distinct_labels):
        d[l] = i
    new_l = [d[l] for l in indices]
    return np.array(new_l)


def accuracy(pred , knoweldge):
    err = pred - knoweldge
    acc = [ ((e==0) and 1 or 0) for e in err]
    return np.sum(acc)/float(pred.shape[0])


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def normalize_data(data, mul=1):
    ss = np.sum(data, -1)
    ss /= mul
    data = data / ss[:,None]
    return data


def zeros_like_pair(l):
    return [np.zeros_like(l[0]), np.zeros_like(l[1])]


class TextHandler:

    def __init__(self, txt_file):
        self.d={}
        self.txt_file = txt_file
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.d[l[:l.index(',')]] = l[l.index(' '):]

    def get_class_txt(self, class_num):
        return self.d[str(class_num)]


def create_ifnot(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def list_files(imfile_list, im_dir):
    with open(imfile_list) as f:
        files = f.readlines()
    files = [im_dir + l.strip('\n') for l in files]
    return file


def min_norm_softmax(x):
    import theano.tensor as T
    e_x = T.exp(x - x.min(keepdims=True))
    return T.nnet.softmax(e_x.reshape((-1, e_x.shape[-1]))).reshape(x.shape)


def max_norm_softmax(x):
    import theano.tensor as T
    e_x = T.exp(x - x.max(keepdims=True))
    return T.nnet.softmax(e_x.reshape((-1, e_x.shape[-1]))).reshape(x.shape)


def temperature_softmax(x):
    if os.environ.get('TT'):
        temperatue = float(os.environ.get('TT'))
    else:
        temperatue =20
    import theano.tensor as T
    e_x = x / temperatue
    return T.nnet.softmax(e_x.reshape((-1, e_x.shape[-1]))).reshape(x.shape)


def get_agnostic_batch(m0, m1, size=256, ratio=.1):
    indices0 = np.random.randint(0, m0.shape[0], (size,))
    tmp_ind = np.random.randint(0, m1.shape[0], (size,))
    # The following code won't function correctly if indices0[i] is zero, but it won't cause a problem
    indices1 = [((np.random.rand() > ratio) and z or indices0[i]) for (i, z) in enumerate(tmp_ind)]
    same = [(indices0[i] == indices1[i]) for i in xrange(size)]
    same = np.array(same).astype('float32')
    return m0[indices0], m1[indices1], same


class AgnosticBatchGen(object):
    def __init__(self, feat, labels, att, size=256, ratio=.1):
        self.size = size
        self.ratio = ratio
        self.labels = labels
        self.feat = feat
        self.att = att
        self.n_instances = self.labels.shape[0]
    def next(self):
        indices0 = np.random.randint(0, self.n_instances, (self.size,))
        tmp_ind = np.random.randint(0, self.n_instances, (self.size,))
        # The following code won't function correctly if indices0[i] is zero, but it won't cause a problem
        indices1 = [((np.random.rand() > self.ratio/2) and z or indices0[i]) for (i, z) in enumerate(tmp_ind)]
        same = (self.labels[indices0]==self.labels[indices1])
        same = np.array(same).astype('float32')
        return {'im':self.feat[indices0],
                'att':self.att[self.labels[indices1]],
                'out':same}

    def get_indices(self, size=None):
        if not size:
            size = self.size
        indices0 = np.random.randint(0, self.n_instances, (size,))
        tmp_ind = np.random.randint(0, self.n_instances, (size,))
        # The following code won't function correctly if indices0[i] is zero, but it won't cause a problem
        indices1 = [((np.random.rand() > self.ratio/2) and z or indices0[i]) for (i, z) in enumerate(tmp_ind)]
        same = (self.labels[indices0]==self.labels[indices1])
        same = np.array(same).astype('float32')
        return indices0, indices1, same


class TripleGenerator(object):
    def __init__(self, feat, labels, att, size=256, ratio=.1):
        self.size = size
        self.ratio = ratio
        self.labels = labels
        self.feat = feat
        self.att = att
        self.n_instances = self.labels.shape[0]

    def next(self):
        indices0 = np.random.randint(0, self.n_instances, (self.size,))
        indices1 = np.random.randint(0, self.n_instances, (self.size,))
        # The following code won't function correctly if indices0[i] is zero, but it won't cause a problem
        same = (self.labels[indices0]==self.labels[indices1])
        same = np.array(same).astype('float32')
        return {'im1': self.feat[indices0],
                'im2': self.feat[indices1],
                'att1': self.att[self.labels[indices0]],
                'att2': self.att[self.labels[indices1]],
                'same': same}

class ThreeImageGen(object):
    def __init__(self, feat, labels, size=256):
        self.size = size
        self.labels = labels
        self.feat = feat
        self.n_instances = self.labels.shape[0]

    def next(self):
        indices = np.random.randint(0, self.n_instances, (self.size,))
        p_ind = [np.random.choice(np.where(self.labels == self.labels[i])[0], 1) for i in indices]
        n_ind = [np.random.choice(np.where(self.labels != self.labels[i])[0], 1) for i in indices]
        p_ind = np.array(p_ind).reshape((self.size, -1))
        n_ind = np.array(n_ind).reshape((self.size, -1))

        return {'im_base': self.feat[indices],
                'im_pos': self.feat[p_ind].reshape((self.size, -1)),
                'im_neg': self.feat[n_ind].reshape((self.size, -1)),
                'out': np.zeros_like(p_ind)
                }


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def save_strlist(fname, l):
    with open(fname, 'a') as f:
        for s in l:
            f.write(s + '\n')

default_w2v = None

def get_word_embedding(word, w2v=None):
    from gensim.models.word2vec import Word2Vec
    global default_w2v
    """
    :param word:
    :return: Word2Vec embedding of image
    !!! returns zero if not in dictionary
    """
    emb_size = 300
    if not w2v:
        if not default_w2v:
            default_w2v = Word2Vec.load_word2vec_format('/home/mohsen/googleNews300.bin.gz', binary=True)
        w2v = default_w2v
    try:
        v = w2v[word]
    except KeyError:
        print('word not in dictionary:%s', word)
        v = np.zeros((emb_size,))
    return v


def arr_except(arr, ind):
    all = np.arange(arr.shape[0])
    desired = np.setdiff1d(all, ind)
    return arr[desired]

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def cluster2vote(clusters, labels, n_cluster=None):
    d = []
    if n_cluster is None:
        n_cluster = np.max(clusters) + 1

    for i in range(n_cluster):
        x = np.where(clusters == i)
        arr = labels[x]
        if arr.size == 0:
            d.append(-1)
            continue
        idx_sort = np.argsort(arr)
        sorted_records_array = arr[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
        d.append(vals[np.argmax(count)])
    pred2vote_byground = np.array([d[p] for p in clusters]) # index out of range error
    return pred2vote_byground


def get_vals_counts(arr):
    idx_sort = np.argsort(arr)
    sorted_records_array = arr[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    return vals, count


def cluster2nobility_opinion(clusters, labels, score, n_clusters=None, nobility_population=.15):
    # score *= -1
    # nobility = np.argsort(score)
    # score *= -1
    pred = -np.ones_like(clusters)
    if n_clusters is None:
        n_clusters = np.max(clusters) + 1
    for k in range(n_clusters):
        ink = np.where(clusters == k)[0]
        if ink.size == 0:
            continue
        k_score = score[ink]
        nobility = np.argsort(k_score)
        nobility = nobility[::-1]
        nobility = nobility[:np.ceil(nobility.shape[0]*nobility_population)]
        opinion = labels[ink[nobility]]
        vals , count = get_vals_counts(opinion)
        pred[ink] = vals[np.argmax(count)]
    return pred

def get_similar_pairs(labels, n_sim, n_nonsim):

    from numpy.random import choice
    sim_indices = np.sort(np.random.randint(labels.shape[0], size=n_sim))
    mates = map(lambda i: choice(np.where(labels == labels[i])[0]), sim_indices)  # TODO where done many times
    sim = (sim_indices, mates)

    # TODO do this in another thread
    unsim_choices = {}
    for l in np.unique(labels):
        unsim_choices[l] = np.where(labels != l)[0]
    nonsim_indices = np.sort(np.random.randint(labels.shape[0], size=n_nonsim))
    mates = np.array(map(lambda i: choice(unsim_choices[labels[i]]), nonsim_indices))
    non_sim = (nonsim_indices,  mates)

    return sim, non_sim


def prediction_confidence(pred, signs, metric='euclidean'):
    dist = cdist(pred, signs, metric=metric)
    margin = np.max(dist)
    min1 = dist.min(1)
    min1ind = dist.argmin(1)
    for i,m in enumerate(min1ind):
        dist[i,m] = (margin+10)

    min2 = dist.min(1)
    score = min2 - min1
    # score = (min1 + 1e-6) ** (-1)
    # score = margin - min1
    return score


def prediction_proximity(pred, signs, metric='euclidean'):
    dist = cdist(pred, signs, metric=metric)
    margin = np.max(dist)
    min1 = dist.min(1)
    score = margin - min1
    return score


def get_class_nobility(score, pred, cat, nb_nob=50):
    z = score.copy()
    z[pred != cat] = 0
    of_cat = np.argsort(z)
    of_cat = of_cat[::-1]
    return of_cat[:nb_nob]


def nob2(clustering, pred, score, n_cluster, n_nob=10):
    n = np.zeros((n_cluster, n_nob), dtype='int')

    for i in range(n_cluster):
        n[i] = get_class_nobility(score, clustering, i, n_nob)

    nv = pred[n]
    y = []
    for row in nv:
        v, c = get_vals_counts(row)
        y.append(v[np.argmax(c)])

    fp = np.zeros_like(pred)
    for i in range(n_cluster):
        fp[clustering==i] = y[i]

    return fp

def mp(arr):
    v,c = get_vals_counts(arr)
    return v[c.argmax()]


def lcm(a, b):
    from fractions import gcd
    return (a*b)//gcd(a, b)

# def get_class_nobility(sorted_arg, pred, cat, nb_nob=50):
#     of_cat = sorted_arg[pred == cat]
#     return of_cat[:nb_nob]


def theano_cdist(X, Y, P):
    import theano
    # import theano.tensor as T
    # X = T.fmatrix('X')
    # Y = T.fmatrix('Y')
    # P = T.scalar('P')
    translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
    minkowski_distances = (abs(translation_vectors) ** P).sum(2) ** (1. / P)
    f_minkowski = theano.function([X, Y, P], minkowski_distances)