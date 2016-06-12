from scipy import misc 
from .utils import readXML

import numpy as np
import tensorflow as tf
import os
import sys


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
        p = self.pos[idx]
        if p == -1:
            if self.ids[self.data_start] != -1:
                kick = self.ids[self.data_start]
                self.pos[kick] = -1
            self.ids[self.data_start] = idx
            self.pos[idx] = self.data_start
            self.data[self.data_start] = misc.imread(self.img_files[idx])
            self.gt[self.data_start] = readXML(self.ano_files[idx]) 
            p = self.data_start
            self.data_start = (self.data_start + 1) % self.size

        # Debug
        # reader = tf.WholeFileReader()
        # path = tf.train.string_input_producer([self.img_files[idx]])
        # _, value = reader.read(path) #self.img_files[idx])
        # sess = tf.Session()

        # print value

        # coord = tf.train.Coordinator()
        # th = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        # print "OK at 45"

        # print "Done at 48"
        # tmp = tf.image.decode_jpeg(value, channels = 3).eval(session = sess)

        # coord.request_stop()

        # coord.join(th)
        # sess.close()
        # print "Now at 52"

        # print tmp.get_shape().as_list()

        # exit()

        img = misc.imread(self.img_files[idx])
        print "Right here at 41"
        # print tmp
        print img.size
        tmp = tf.Variable(img)
        print tmp.get_shape().as_list(), tmp.dtype
        exit()
        #
        return self.gt[p], self.data[p]

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

    # directory_scan
    def _scan_dir(self, name, img_path, ano_path):
        img_files = []
        for dir_path, _, dir_files in os.walk(img_path):
            for f in dir_files:
                img_files.append(os.path.join(dir_path, f))
        ano_files = []
        for dir_path, _, dir_files in os.walk(ano_path):
            for f in dir_files:
                ano_files.append(os.path.join(dir_path, f))

        self.data[name] = Pool(img_files, ano_files, self.pool_size)

    def get_data(self, name, idx):
        return self.data[name].query(idx)
