import scipy.io
import numpy as np
import h5py
import pandas as pd


def load_mat():
    mat = scipy.io.loadmat('annotations.mat')
    yo = mat['None']
    print(yo)

def load_mat2():
    f = h5py.File('somefile.mat', 'r')
    data = f.get('data/annotations')
    data = np.array(data)
    print(data)

def load_mat3():
    mat = scipy.io.loadmat('annotations.mat')
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})  # compatible for both python 2.x and python 3.x

    data.to_csv("example.csv")


load_mat()

