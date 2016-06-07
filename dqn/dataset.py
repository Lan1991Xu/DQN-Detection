import numpy as np
import tensorflow as tf
import os
import sys

from scipy import misc
from .utils import readXML

class Pool(object):
    def __init__(self, img_files, ano_files, size):
        self.size = size
        self.img_files = np.array(img_files)
        self.ano_files = np.array(ano_files)
        self.data = np.empty(size, dtype = np.ndarray) 
        self.gt = np.empty(size, dtype = np.ndarray)
        self.pos = np.zeros(size, dtype = int)
        self.ids = np.zeros(size, dtype = int)
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
            self.data[self.data_start] = misc.imread(self.img_files[idx])
            self.gt[self.data_start] = readXML(self.ano_files[idx]) 
            self.data_start = (self.data_start + 1) % self.size
        return self.gt[idx], self.data[idx]

class Dataset(object):
    def __init__(self, train_dir, train_ano_dir, test_dir = None, test_ano_dir = None, pool_size = 1000):
        self.tr_dir = train_dir
        self.tr_ano_dir = train_ano_dir
        self.te_dir = test_dir
        self.te_ano_dir = test_ano_dir
        self.pool_size = pool_size

        self.data = {}

        self._scan_dir('train', self.tr_dir, self.tr_ano_dir)
        if self.te_dir != None:
            self._scan_dir('test', self.te_dir, self.te_ano_dir)

    # self_scan
    def _scan_dir(self, name, img_path, ano_path):
        for dir_path, _, dir_files in os.walk(img_path):
            img_files = dir_files

        for dir_path, _, dir_files in os.walk(ano_path):
            ano_files = dir_files

        self.data[name] = Pool(img_files, ano_files, self.pool_size)

    def get_data(name, idx):
        return self.data[name].query(idx)
