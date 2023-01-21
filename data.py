import idx2numpy
import numpy as np
import os
import pickle

def load_data(data_directory, train = True):
    if train:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_labels'))
    else:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_labels'))

    vdim = images.shape[1] * images.shape[2]
    vectors = np.empty([images.shape[0], vdim])
    for imnum in range(images.shape[0]):
        imvec = images[imnum, :, :].reshape(vdim, 1).squeeze()
        vectors[imnum, :] = imvec
    
    return vectors, labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    uc = X.mean()
    sdc = X.std()
    if u is not None:
        uc = u
    if sd is not None:
        sdc = sd
    X = (X-uc)/sdc
    return (X, uc, sdc)

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    minc = X.min()
    maxc = X.max()
    if _min is not None:
        minc = _min
    if _max is not None:
        maxc = _max
    X = (X-minc)/(maxc-minc)
    return (X, minc, maxc)

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    k = 10
    ohe = []
    for l in y:
        ohe.append(np.eye(k)[l-1])
    return np.array(ohe)

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)+1

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    Images, Labels = dataset
    n = len(Images)
    idx = np.random.permutation(n)
    return (Images[idx], Labels[idx])

def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d)
    """
    _, d = X.shape
    return np.vstack((X, np.zeros((1, d))))

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 5): 
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width
