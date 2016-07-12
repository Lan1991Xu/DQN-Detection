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

    def query(self, sess, target_class):
        self.gt = readXML(self.ano_files[self.data_pos], target_class) 
        self.data_pos += 1
        self.img.eval(session = sess)

        return self.gt, self.img

class Dataset(object):
    def __init__(self, target_class, train_list, img_dir, ano_dir, test_list = None, tot_epoches = 1):
        self.train_list = train_list
        self.test_list = test_list
        self.ano_dir = ano_dir
        self.img_dir = img_dir
        self.tot_epoches = tot_epoches
        self.target_class = target_class

        self.data = {}

        self._get_list('train', self.img_dir, self.ano_dir, self.train_list, self.tot_epoches)
        if self.test_list != None:
            self._get_list('test', self.img_dir, self.ano_dir, self.test_list, 1)

    # directory_scan
    def _get_list(self, name, img_dir, ano_dir, img_list,rep):
        img_files = []
        ano_files = []
        # for dir_path, _, dir_files in os.walk(img_path):
        #     for f in dir_files:
        #         img_files.append(os.path.join(dir_path, f))
        list_input = open(img_list, 'r')
        for line in list_input.readlines():
            sep = line.strip().split(' ')
            if(sep[-1] != '-1'):
                img_files.append(os.path.join(img_dir, sep[0] + '.jpg'))
                ano_files.append(os.path.join(ano_dir, sep[0] + '.xml'))
        list_input.close()

        # for dir_path, _, dir_files in os.walk(ano_path):
        #     for f in dir_files:
        #         ano_files.append(os.path.join(dir_path, f))

        self.data[name] = Pool(img_files, ano_files, rep)

    def get_data(self, name, sess):
        return self.data[name].query(sess, self.target_class)

    def get_size(self, name):
        return self.data[name].size
