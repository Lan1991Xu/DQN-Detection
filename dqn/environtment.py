import tensorflow as tf
import numpy as np
import os

from .dataset import Dataset
from config import Config

class State(object):
    def __init__(self, img, height, width):
        self.img = img
        self.height, self.width = width 
        # box = [top, left, down, right]
        self.box = [1, 1, self.height, self.width]  

    def clip_box(self):
        self.box[0] = max(self.box[0], 1)
        self.box[1] = max(self.box[1], 1)
        self.box[2] = min(self.box[2], self.height)
        self.box[3] = min(self.box[3], self.width)

class Environment(object):
    def __init__(self, config, sess):
        self.data = Dataset(config.train_dir, config.train_ano_dir, config.test_dir, config.test_ano_dir, config.pool_size)
        self.cur_img = 0
        self.alpha = config.alpha 
        self.state = None
        self.action_size = 8
        self.IoU = 0.0
        self.accept_rate = config.accept_rate
        self.eps = config.eps
        self.define_act()
        self.sess = sess
    
    def _act(self, action):
        self.move[str(action)]()
        self.state.clip_box()
        self._calc_IoU()
    def _calc_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    def _isIntersect(self, box, gt):
        if box[0] >= gt[2]:
            return False
        if box[2] <= gt[0]:
            return False
        if box[1] >= gt[3]:
            return False
        if box[3] <= gt[1]:
            return False
        return True
    def _calc_IoU(self):
        gt = self.ground_truth
        box = self.state.box
        if self._isIntersect(box, gt):
            inter = [max(box[0], gt[0]), max(box[1], gt[1]), min(box[2], gt[2]), min(box[3], gt[3])]
            interArea = self._calc_area(inter)
            self.IoU = (1. * interArea) / (self._calc_area(box) + self._calc_area(gt) - interArea + 0.)
        else:
            self.IoU = 0.0

    def define_act(self):
        self.move = {}
        self.move['0'] = self.move_left
        self.move['1'] = self.move_right
        self.move['2'] = self.move_up
        self.move['3'] = self.move_down
        self.move['4'] = self.bigger
        self.move['5'] = self.smaller
        self.move['6'] = self.fatter
        self.move['7'] = self.taller

    def move_left(self):
        self.state.box[1] -= self.alpha
        self.state.box[3] -= self.alpha
    def move_right(self):
        self.state.box[1] += self.alpha
        self.state.box[3] += self.alpha
    def move_up(self):
        self.state.box[0] -= self.alpha
        self.state.box[2] -= self.alpha
    def move_down(self):
        self.state.box[0] += self.alpha
        self.state.box[2] += self.alpha
    def bigger(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        delta_x = int(delta_x)
        delta_y = int(delta_y)
        self.state.box[0] -= delta_x
        self.state.box[2] += delta_x
        self.state.box[1] -= delta_y
        self.state.box[3] += delta_y
    def smaller(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        delta_x = int(delta_x)
        delta_y = int(delta_y)
        self.state.box[0] += delta_x
        self.state.box[2] -= delta_x
        self.state.box[1] += delta_y
        self.state.box[3] -= delta_y
    def fatter(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        delta_x = int(delta_x)
        self.state.box[0] += delta_x
        self.state.box[2] -= delta_x
    def taller(self):
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        delta_y = int(delta_y)
        self.state.box[1] += delta_y
        self.state.box[3] -= delta_y

    def clear(self):
        self.cur_img = 0
    def reset(self, isTrain = True):
        if isTrain:
            self.ground_truth, pic = self.data.get_data('train', self.cur_img, self.sess)
            #
            print type(pic)
            exit()
            #
            self.state = State(pic, pic.shape[0], pic.shape[1])
            self.cur_img = (self.cur_img + 1) % self.train_size
        else:
            self.ground_truth, pic = self.data.get_data('test_img', self.cur_img)
            self.state = State(pic, pic.shape[0], pic.shape[1])
            self.cur_img += 1

        return self.state

    def _sign(self, x):
        return 1 if x >= self.eps else -1
    def _isTerminal(self):
        if self.IoU >= self.accept_rate:
            return True
        else:
            return False

    def act(self, action):
        pre_IoU = self.state.IoU
        self._act(action)
        return self.state, self._sign(self.IoU - pre_IoU), _isTerminal()
