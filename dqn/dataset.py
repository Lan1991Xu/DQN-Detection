import numpy as np
import tensorflow as tf
import os
import sys

from scipy import misc

# needed train_img, train_  
class Dataset(object):
    def __init__(self, train_dir, train_ano_dir, test_dir, test_ano_dir, pool_size):
        self.tr_dir = train_dir
        self.tr_ano_dir = train_ano_dir
        self.te_dir = test_te_dir
        self.te_ano_dir = test_ano_dir
        self.pool_size = pool_size

        self.data = {}

        self.data['train_img'] = Pool()
        self.data['train_ano'] = Pool()
        self.data['test_img'] = Pool()
        self.data['test_ano'] = Pool()

    # self_scan
    def _scan_dir(self, path, pl):
        

    def get_data(name, idx):
