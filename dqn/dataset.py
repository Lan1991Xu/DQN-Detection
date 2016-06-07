import numpy as np
import tensorflow as tf
import os
import sys

from scipy import misc

class Pool(object):
    def __init__(self, files, size):
        self.size = size
        self.files = np.array(files)
        self.data = np.empty(size, dtype = ndarray) 
        self.pos = np.zero(size, dtype = int)
        self.ids = np.zero(size, dtype = int)
        self.ids -= 1
        self.pos -= 1
        self.data_start = 0

    def query(self, idx):
        if self.pos[idx] != -1:
            if self.ids[self.data_start] != -1:
                kick = self.ids[self.data_start]
                self.pos[kick] = -1
            self.ids[self.data_start] = idx
            self.pos[idx] = self.data_start
            self.data[self.data_start] = misc.imread(self.files[idx])
            self.data_start = (self.data_start + 1) % self.size
        return self.data[idx]

class Dataset(object):
    def __init__(self, train_dir, train_ano_dir, test_dir, test_ano_dir, pool_size):
        self.tr_dir = train_dir
        self.tr_ano_dir = train_ano_dir
        self.te_dir = test_te_dir
        self.te_ano_dir = test_ano_dir
        self.pool_size = pool_size

        self.data = {}

        self._scan_dir('train_img', self.tr_dir)
        self._scan_dir('train_ano', self.tr_ano_dir)
        self._scan_dir('test_img', self.te_dir)
        self._scan_dir('test_ano', self.te_ano_dir)

    # self_scan
    def _scan_dir(self, name, path):
        files = []
        for dir_path, _, dir_files in os.walk(path):
            for f in dir_files:
                files.append(dir_path + f)
        self.data[name] = Pool(files, self.pool_size)

    def get_data(name, idx):
        return self.data[name].query(idx)
