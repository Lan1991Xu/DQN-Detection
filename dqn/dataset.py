from .utils import readXML

import numpy as np
import tensorflow as tf
import os
import sys


class Pool(object):
    def __init__(self, img_files, ano_files, rep):
        self.img_files = np.tile(np.array(img_files), rep)
        self.ano_files = np.tile(np.array(ano_files), rep)
        self.data_pos = 0
        self.reader = tf.WholeFileReader()
        path_queue = tf.train.string_input_producer(img_files, shuffle = False)
        _, value = self.reader.read(path_queue)
        self.img = tf.image.decode_jpeg(value, channels = 3)
        self.size = self.img_files.shape[0]

    def query(self, sess):
        self.gt = readXML(self.ano_files[self.data_pos]) 
        self.data_pos += 1
        self.img.eval(session = sess)

        return self.gt, self.img

class Dataset(object):
    def __init__(self, train_dir, train_ano_dir, test_dir = None, test_ano_dir = None, tot_epoches = 1):
        self.tr_dir = train_dir
        self.tr_ano_dir = train_ano_dir
        self.te_dir = test_dir
        self.te_ano_dir = test_ano_dir
        self.tot_epoches = tot_epoches

        self.data = {}

        self._scan_dir('train', self.tr_dir, self.tr_ano_dir, self.tot_epoches)
        if self.te_dir != None:
            self._scan_dir('test', self.te_dir, self.te_ano_dir, 1)

    # directory_scan
    def _scan_dir(self, name, img_path, ano_path, rep):
        img_files = []
        for dir_path, _, dir_files in os.walk(img_path):
            for f in dir_files:
                img_files.append(os.path.join(dir_path, f))
        ano_files = []
        for dir_path, _, dir_files in os.walk(ano_path):
            for f in dir_files:
                ano_files.append(os.path.join(dir_path, f))

        self.data[name] = Pool(img_files, ano_files, rep)

    def get_data(self, name, sess):
        return self.data[name].query(sess)

    def get_size(self, name):
        return self.data[name].size
