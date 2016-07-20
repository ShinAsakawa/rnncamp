"""
A SRN (Simple recurrent neural network), a.k.a Elman net, for RNN camp
All the copyrights belong to Shin Asakawa <asakawa@ieee.org>

N.B.: Please confirm your python can understand the termcolor.
If not, please `pip` it before you run me.
```
pip install termcolor
```
"""

from __future__ import absolute_import
from __future__ import print_function
import six
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from termcolor import colored, cprint

### Definition of global variables, such as hyperparameters
hidden_size = 100    # size of hidden layer of neurons
lr          = 0.01   # learning ratio
max_iter    = 10000  # maximam iterations
sample_n    = 100    # number of words for prediction in snapshot
seq_length  = 25     # number of steps to unroll the RNN for
snapshot_t  = 100    # interval of shapshot
grad_clip   = 1.0    # threshold for grad_clilp

parser = argparse.ArgumentParser()

parser.add_argument('--activate_f', '-activate_f', default='tanh', \
                    help='[tanh|logistic|relu|elu]')
parser.add_argument('--grad_clip', '-grad_clip', type=float, default=1.0, \
                    help='threshold value for gradient clipping')
parser.add_argument('--hidden', '-hidden', type=int, default=100, \
                    help='number of neurons in hidden layer')
parser.add_argument('--lr', '-lr', type=float, default=0.01, \
                    help='learning ratio')
parser.add_argument('--max_iter', '-max_iter', type=int, default=10000)
parser.add_argument('--sample_n', '-sample_n', type=int, default=50, \
                    help='number of items to predict')
parser.add_argument('--seed', '-seed', type=int, default=1,
                    help='seed (initial value) for random number generator')
parser.add_argument('--seq_length', '-seq_length', type=int, default=25)
parser.add_argument('--snapshot_t', '-snapshot_t', type=int, default=100, \
                    help='snapshot interval')
parser.add_argument('--train', '-train', default=None, \
                    help='text file name for training')

### preparation to draw graphs
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def make_dataset(filename):
    """Loads a tex file as data file, and return line into an encoded sequence.
    """
    data = []

    # file to be read must be a simple plain text file
    with open(filename, "r") as f:
        content = f.readlines()

    ## remove blank lines
    content = [line for  line in content if len(line) > 2]
    for line in content:
        for w in line.strip().split():
            data.append(w)
        data.append('</s>')   ## w.r.t Tomas Mikolov(2011, 2013)

    vocab = list(set(data))
    data_size, vocab_size = len(data), len(vocab)
    print("data has %d words among %d unique." % (data_size, vocab_size))

    for w in data:
        if w == '</s>':
             print()
        else:
            print("%s" % w, end=' ')
    return data, vocab

### definition of activation fuctions
###
## a.k.a sigmoid function.
## However, I do not like to call this a sigmoid function
def logistic(x):
    one = x[0].dtype.type(1)
    half = x[0].dtype.type(0.5)
    return half * (np.tanh(x) + one)
    ##return 1.0/(1.0 + np.exp(-x))

def logistic_grad(x, delta):
    one = x[0].dtype.type(1)
    return x * (one - x) * delta

def tanh(x):
    return np.tanh(x)

def tanh_grad(x, delta):
    one = x[0].dtype.type(1)
    return (one - x * x) * delta

def relu(x):
    return np.maximum(x, 0, x)

def relu_grad(x, delta):
    tmp = [1.0 if a > 0 else 0.0 for a in x[0]]
    return (tmp * delta)

### elu (see arXiv:1511.07289v5 [cs.LG]) means an exponential linear units
def elu(x):
    y = x.copy()
    neg_indecies = x < 0
    y[neg_indecies] = (np.exp(y[neg_indecies]) - 1)
    return y

def elu_grad(x, delta):
    tmp = [np.exp(a) if a <= 0 else 1.0 for a in x[0]]
    return (tmp * delta)


def lossFun(X, Y, H_prev, vocab_size, \
            W_xh, W_hh, W_hy, b_y, b_h, \
            grad_clip, activate_f):
    """
    X (input), Y (target) are both list of integers.
    H_prev is H x 1 array of initial hidden state
    returns the loss, gradients on model parameters, last hidden state,
    and an array for plot gradients
    """
    X_s, H_s, Y_s, Prb = {}, {}, {}, {}
    H_s[-1] = np.copy(H_prev)

    loss = 0
    ## forward pass
    for t in xrange(len(X)):
        # encode one hot vector
        X_s[t] = np.zeros((vocab_size, 1))
        X_s[t][X[t]] = 1

        # calc hidden state
        state = np.dot(W_xh, X_s[t]) + np.dot(W_hh, H_s[t - 1]) + b_h
        #H_s[t] = np.tanh(np.dot(W_xh, X_s[t]) + np.dot(W_hh, H_s[t - 1]) + b_h)
        #H_s[t] = tanh(np.dot(W_xh, X_s[t]) + np.dot(W_hh, H_s[t - 1]) + b_h)
        if activate_f == 'tanh':
            H_s[t] = tanh(state)
        elif activate_f == 'logistic':
            H_s[t] = logistic(state)
        elif activate_f == 'relu':
            H_s[t] = relu(state)
        else:
            H_s[t] = elu(state)

        # unnormalized log probabilities for next chars
        Y_s[t] = np.dot(W_hy, H_s[t]) + b_y

        # probabilities for next items
        # softmax (cross-entropy loss)
        Prb[t] = np.exp(Y_s[t]) / np.sum(np.exp(Y_s[t]))
        loss += -np.log(Prb[t][Y[t], 0])

    # backward pass: compute gradients going backwards
    dW_xh   = np.zeros_like(W_xh)
    dW_hh   = np.zeros_like(W_hh)
    dW_hy   = np.zeros_like(W_hy)
    db_h    = np.zeros_like(b_h)
    db_y    = np.zeros_like(b_y)
    dh_next = np.zeros_like(H_s[0])

    for t in reversed(xrange(len(X))):
        dy = np.copy(Prb[t])

        # backpropagation into the output vector y
        dy[Y[t]] -= 1
        dW_hy += np.dot(dy, H_s[t].T)
        db_y  += dy

        # backpropagation into the hidden layer h
        dh = np.dot(W_hy.T, dy) + dh_next

        # backpropation through nonlinearity tanh()
        #dh_raw  =  (1.0 - H_s[t] * H_s[t]) * dh
        if activate_f == 'tanh':
            dh_raw = tanh_grad(H_s[t], dh)
        elif activate_f == 'logistic':
            dh_raw = logistic_grad(H_s[t], dh)
        elif activate_f == 'relu':
            dh_raw = relu_grad(H_s[t], dh)
        else:
            dh_raw = elu_grad(H_s[t], dh)

        # print(H_s[t].shape)
        # print('----------')
        # print(dh_raw.shape)
        # sys.exit()

        db_h    += dh_raw
        dW_xh   += np.dot(dh_raw, X_s[t  ].T)
        dW_hh   += np.dot(dh_raw, H_s[t-1].T)
        dh_next =  np.dot(W_hh.T, dh_raw)

    ##Gradient clipping to avoid gradient exploding problem
    for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
        np.clip(dparam, -grad_clip, grad_clip, out=dparam)

    return loss, dW_xh, dW_hh, dW_hy, db_h, db_y, H_s[len(X) - 1]

def make_W_matrices(vocab_size):
    W_xh = np.random.randn(hidden_size, vocab_size)  * 0.01  # input to hidden
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    W_hy = np.random.randn(vocab_size,  hidden_size) * 0.01  # hidden to output
    b_h  = np.zeros((hidden_size, 1))                        # hidden bias
    b_y  = np.zeros((vocab_size, 1))                         # output bias
    return W_xh, W_hh, W_hy, b_h, b_y


def sample(h, seed_index, n, vocab, W_xh, W_hh, W_hy, b_h, b_y, vocab_size):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    ## Make a one-hot-vector
    x = np.zeros((vocab_size, 1))
    x[seed_index] = 1

    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h)
        y = np.dot(W_hy, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def print_snapshot(H_prev, inputs, sample_n, vocab, W_xh, W_hh, W_hy, b_h, b_y, vocab_size, index_to_word):
    sample_index = sample(H_prev, inputs[0], sample_n, vocab, W_xh, W_hh, W_hy, b_h, b_y, vocab_size)

    print('-----')
    for i in sample_index:
        if index_to_word[i] == '</s>':
            print()
        else:
            print('%s' % index_to_word[i], end=' ')
    print('\n-----')


def trainNet(params, data, vocab):
    ## Retrieve parameters
    activate_f  = params['activate_f']
    grad_clip   = params['grad_clip']
    hidden_size = params['hidden_size']
    lr          = params['lr']
    max_iter    = params['max_iter']
    sample_n    = params['sample_n']
    seq_length  = params['seq_length']
    snapshot_t  = params['snapshot_t']
    train_file  = params['train_file']

    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    ## Make weight matrices
    W_xh, W_hh, W_hy, b_h, b_y = make_W_matrices(vocab_size)
    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), \
                          np.zeros_like(W_hh), \
                          np.zeros_like(W_hy)

    ## Memory variables for Adagrad
    mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y)

    # an array for plotting gradients
    graph_W_xh, graph_W_hh, graph_W_hy = [], [], []
    graph_bh, graph_by = [], []
    loss_array = []


    ## Loss at iteration 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length

    ## main loop for training
    pointer = 0
    for iter in xrange(max_iter):
        ## Preparaton for inputs
        ## Sweeping from left to right in steps seq_length long
        if pointer + seq_length + 1 >= len(data) or iter == 0:
            # reset RNN memory
            H_prev = np.zeros((hidden_size, 1))

            # go from start of data
            pointer = 0

        X = [word_to_index[w] for w in data[pointer  :pointer+seq_length  ]]
        Y = [word_to_index[w] for w in data[pointer+1:pointer+seq_length+1]]

        # sample from the model now and then
        if iter % snapshot_t == 0:
            print_snapshot(H_prev, X, sample_n, \
                           vocab, W_xh, W_hh, W_hy, b_h, b_y, \
                           vocab_size, index_to_word)


        # forward seq_length characters through the net and fetch gradient
        loss, dW_xh, dW_hh, dW_hy, db_h, db_y, H_prev = \
            lossFun(X, Y, H_prev, vocab_size, \
                    W_xh, W_hh, W_hy, b_y, b_h, grad_clip, activate_f)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        loss_array.append(loss)
        graph_W_xh.append(np.sum(np.abs(dW_xh)) / np.sum(np.abs(dW_xh.shape)))
        graph_W_hh.append(np.sum(np.abs(dW_hh)) / np.sum(np.abs(dW_hh.shape)))
        graph_W_hy.append(np.sum(np.abs(dW_hy)) / np.sum(np.abs(dW_hy.shape)))
        graph_bh.append  (np.sum(np.abs(db_h )) / np.sum(np.abs(db_h.shape )))
        graph_by.append  (np.sum(np.abs(db_y )) / np.sum(np.abs(db_y.shape )))

        # print progress
        if iter % snapshot_t == 0:
            #print('iter %d, loss: %f' % (iter, smooth_loss))
            cprint('iter'     , 'green', end=' ')
            cprint(iter       , 'green', end=' ')
            cprint(' loss:'   , 'green', end=' ')
            cprint(smooth_loss, 'green')

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([ W_xh,  W_hh,  W_hy,  b_h,  b_y], \
                                      [dW_xh, dW_hh, dW_hy, db_h, db_y], \
                                      [mW_xh, mW_hh, mW_hy, mb_h, mb_y]):
            # adagrad update
            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + 1e-8)

        # move data pointer
        pointer += seq_length
    return loss_array, graph_W_xh, graph_W_hh, graph_W_hy, graph_bh, graph_by


def main(params):
    ## Make dataset
    data, vocab = make_dataset(train_file)

    loss_array, garray_dW_xh, garray_dW_hh, garray_dW_hy, \
        garray_dbh, garray_dby = trainNet(params, data, vocab)

    plt.plot(np.array(loss_array))
    plt.title("loss values as a function of iteration")
    plt.show()
    plt.clf()

    plt.plot(np.array(garray_dW_xh))
    plt.plot(np.array(garray_dW_hh))
    plt.plot(np.array(garray_dW_hy))
    plt.plot(np.array(garray_dbh))
    plt.plot(np.array(garray_dby))
    plt.legend(('dW_xh', 'dW_hh', 'dW_hy', 'db_h', 'db_y'))
    plt.title("sum of gradients as a function of iteration")
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    ## set random seed
    if args.seed:
        rand_seed = args.seed
        print("Rand seed: %d" % rand_seed)
        np.random.seed(rand_seed)
    else:
        np.random.seed(1)

    ## set activate_f
    if args.activate_f:
        activate_f = args.activate_f
        if not activate_f == 'tanh' \
           and not activate_f == 'logistic' \
           and not activate_f == 'relu' \
           and not activate_f == 'elu':
            print('Unknown activation function: %s' % activate_f)
            sys.exit()
        print("Activatation function: %s" % activate_f)

    ## set grad_clip
    if args.grad_clip:
        grad_clip = args.grad_clip
        print("Grad clip: %f" % grad_clip)
    else:
        grad_clip = 5.0

    ## set hidden_size
    if args.hidden:
        hidden_size = args.hidden
        print("Hidden size: %d" % hidden_size)

    ## set learning ratio
    if args.lr:
        lr = args.lr
        print("lr: %f (learning ratio)" % lr)

    ## set max_iter
    if args.max_iter:
        max_iter = args.max_iter
        print("Max_iter: %d" % max_iter)

    ## set sample_n
    if args.sample_n:
        sample_n = args.sample_n
        print("Sample_n: %d" % sample_n)

    ## set seq_length
    if args.seq_length:
        seq_length = args.seq_length
        print("Seq_length: %f" % seq_length)

    ## set snapshot_t
    if args.snapshot_t:
        snapshot_t = args.snapshot_t
        print("Snapshot_t: %d" % snapshot_t)

    ## set train file
    if args.train:
        train_file = args.train
        print("Train filename: %s" % train_file)
    else:
        train_file = __file__

    params = {
        'activate_f': activate_f,
        'grad_clip' : grad_clip,
        'hidden_size': hidden_size,
        'lr': lr,
        'max_iter': max_iter,
        'sample_n': sample_n,
        'seq_length': seq_length,
        'snapshot_t': snapshot_t,
        'train_file': train_file,
    }
    main(params)
