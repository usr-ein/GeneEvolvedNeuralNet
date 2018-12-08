import numpy as np
# Fast sigmoid, slower for single value cause C overhead
from scipy.special import expit

def linear(x):
    return x

# def sigmoid(x):
#     return expit(x)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x < 0] = 0
    return x

def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def elu(x, a=2):
    '''exponential linear unit, from this paper  https://arxiv.org/abs/1511.07289... seems to work quite well'''
    return np.where(x <= 0, a * (np.exp(x) - 1), x)

def gaussian(x):
    return np.exp(np.negative(np.square(x)))

def rmse(pred, targets):
    return np.sqrt(((pred - targets)**2).mean()).astype('float32')

def random_derangement(n):
    ''' Random permutations without fixed points a.k.a. derangement.
    Parameters
    ----------
        n : int
            Size of the range of integers to find the random derangement of.

    Notes
    -----
    About 40% slower than np.random.permutation but that's not much
    59.4 µs ± 4.13 µs
    See https://stackoverflow.com/a/52614780/5989906
    '''
    original = np.arange(n)
    new = np.random.permutation(n)
    same = np.where(original == new)[0]
    while same.size > 0:  # while not empty
        swap = same[np.random.permutation(len(same))]
        new[same] = new[swap]
        same = np.where(original == new)[0]
        if len(same) == 1:
            swap = np.random.randint(0, n)
            new[[same[0], swap]] = new[[swap, same[0]]]
    return new
def normalTrucatedMultiple(n, size=1):
    ''' Return size samples of the normal distribution around
    n/2 truncated so that min is 0 and max is n-1.

    Parameters
    ----------
    n : int
        Parameter of the distribution according to desc.
    size: int, optional
        Output shape.
        If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        If size is None (default), a single value is returned if loc and scale are both scalars.
        Otherwise, np.broadcast(loc, scale).size samples are drawn.

    Notes
    -----
    For one sample:
    3.8 µs ± 839 ns
    But for size=5000 samples:
    129 µs ± 1.05 µs
    '''

    return np.random.normal(
        n * 0.5, n * 0.33, size=size).astype(int).clip(0, n - 1)


def vectorize(func):
    def vfunc(x):
        y = []
        for xi in x:
            y.append(func(xi))
        return np.array(y)
    return vfunc
