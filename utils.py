import requests
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer


def lire_alpha_digit(focus_list, path=None):
    if path == None: 
        alphadigs_url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
        r = requests.get(alphadigs_url, allow_redirects=True)
        filename = 'binaryalphadigs.mat'
        open('../data/' + filename, 'wb').write(r.content)
        path = '../data/binaryalphadigs.mat'
        print('Download completed, data available in /data/binaryalphadigs.mat')
    
    elif os.path.exists(path):
        print('Path correct')

    data_dic = loadmat(path)

    digit2idx = {}
    for i, digit in enumerate(data_dic['classlabels'][0]):
        digit2idx[digit[0]] = i

    focus_idx =[]
    for digit in focus_list: 
        focus_idx.append(digit2idx[digit])
    
    data = np.stack(np.concatenate(data_dic['dat'][focus_idx])).reshape(-1, 16*20)
    print(data)
    return data 


"""def load_mnist():
    list_name = ['train-images', 'train-labels', 't10k-images', 't10kimages']
    for name in list_name: 
        mnist_url = f'http://yann.lecun.com/exdb/mnist/{name}-idx3-ubyte.gz'
        r = requests.get(mnist_url, allow_redirects=True)
        filename = f'mnist-{name}.gz'
        open('data/' + filename, 'wb').write(r.content)
        print(f'Download completed, data available in /data/{name}-idx3-ubyte.gz')"""

def fetch_alpha_digits_data():
    alphadigs_url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
    r = requests.get(alphadigs_url, allow_redirects=True)
    filename = 'binaryalphadigs.mat'
    open('data/' + filename, 'wb').write(r.content)
    
def fetch_mnist_digits_data(data_length):
    # Fetch data 
    mnist = fetch_openml('mnist_784')

    # Get pixels values
    X = mnist['data'].values.copy() #apply(lambda x: 1 if x > 125 else 0)
    # From grayscale to black and white
    X[X < 125] = 0 
    X[X >= 125] = 1

    #train_length = (int)(data_length*test_train)

    X_train, X_test = X[:data_length, :], X[data_length:data_length+10000, :]
    # get labels 
    y = mnist['target'].values.copy()
    y_train, y_test = y[:data_length], y[data_length:data_length+10000]
    # One hot encoding for labels 
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    return X_train, X_test, y_train, y_test

def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def plot_examples_alphadigits(X_train, x_generated, nb_iterations, outputpath = '../images/'):
    """
    takes as input X_train (alphadigits stacked into rows) and x_generated a tensor of dimensiosn n_images x 20 x 16
    plots a comparaison between generated and training examples..
    """
    fig = plt.figure(figsize=(8, 8))
    n_generated = x_generated.shape[0]
    columns = min(n_generated, 4)
    rows = 2
    samples = np.random.choice(np.arange(X_train.shape[0]), size=columns, replace=False)