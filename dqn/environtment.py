import tensorflow as tf
import numpy as np
import os

from .dataset import Dataset
from config import Config

class State(object):
    def __init__(self, img, height, width):
        self.img = img
        self.height, self.width = height + 0., width + 0. 
        # box = [top, left, down, right]
        self.box = [1., 1., self.height, self.width]  

    def clip_box(self):
        self.box[0] = max(self.box[0], 1.)
        self.box[1] = max(self.box[1], 1.)
        self.box[2] = min(self.box[2], self.height)
        self.box[3] = min(self.box[3], self.width)

class Environment(object):
    def __init__(self, config, sess):
        self.data = Dataset(config.train_list, config.img_dir, config.ano_dir, config.test_list, config.tot_epoches)
        # self.cur_img = 0
        self.alpha = config.alpha # The rescale step rate.
        self.move_alpha = config.move_alpha # The movement step rate

        self.state = None
        self.action_size = config.action_size
        self.tri_reward = config.trigger_reward
        self.tri_thres = config.trigger_threshold
        self.IoU = 0.0
        self.accept_rate = config.accept_rate
        self.eps = config.eps
        self.define_act()
        self.sess = sess
        self.train_size = self.data.get_size('train')
        self.test_size = self.data.get_size('test')
    
    def start(self):
        self.coord = tf.train.Coordinator()
        self.thread = tf.train.start_queue_runners(coord = self.coord, sess = self.sess)

    def _act(self, action):
        self.move[action]()
        self.state.clip_box()
        self._calc_IoU()
    def _calc_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    def _isIntersect(self, box, gt):
        if self._sign(box[0] - gt[2]) == 1:
            return False
        if self._sign(gt[0] - box[2]) == 1:
            return False
        if self._sign(box[1] - gt[3]) == 1:
            return False
        if self._sign(gt[1] - box[3]) == 1:
            return False
        return True
    def _calc_IoU(self):
        gt = self.ground_truth
        box = self.state.box
        if self._isIntersect(box, gt):
            inter = [max(box[0], gt[0]), max(box[1], gt[1]), min(box[2], gt[2]), min(box[3], gt[3])]
            interArea = self._calc_area(inter)
            self.IoU = interArea / (self._calc_area(box) + self._calc_area(gt) - interArea)
        else:
            self.IoU = 0.0

    def define_act(self):
        self.move = [] 
        self.move.append(self.move_left)
        self.move.append(self.move_right)
        self.move.append(self.move_up)
        self.move.append(self.move_down)
        self.move.append(self.bigger)
        self.move.append(self.smaller)
        self.move.append(self.fatter)
        self.move.append(self.taller)

    def move_left(self):
        stp_size = (self.state.box[3] - self.state.box[1]) * self.move_alpha
        self.state.box[1] -= stp_size
        self.state.box[3] -= stp_size 
    def move_right(self):
        stp_size = (self.state.box[3] - self.state.box[1]) * self.move_alpha
        self.state.box[1] += stp_size 
        self.state.box[3] += stp_size 
    def move_up(self):
        stp_size = (self.state.box[2] - self.state.box[0]) * self.move_alpha
        self.state.box[0] -= stp_size 
        self.state.box[2] -= stp_size
    def move_down(self):
        stp_size = (self.state.box[2] - self.state.box[0]) * self.move_alpha
        self.state.box[0] += stp_size
        self.state.box[2] += stp_size
    def bigger(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        self.state.box[0] -= delta_x
        self.state.box[2] += delta_x
        self.state.box[1] -= delta_y
        self.state.box[3] += delta_y
    def smaller(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        self.state.box[0] += delta_x
        self.state.box[2] -= delta_x
        self.state.box[1] += delta_y
        self.state.box[3] -= delta_y
    def fatter(self):
        delta_x = ((self.state.box[0] + self.state.box[2]) * 0.5 - self.state.box[0]) * self.alpha
        self.state.box[0] += delta_x
        self.state.box[2] -= delta_x
    def taller(self):
        delta_y = ((self.state.box[1] + self.state.box[3]) * 0.5 - self.state.box[1]) * self.alpha
        self.state.box[1] += delta_y
        self.state.box[3] -= delta_y
    def trigger_reward(self):
        if self.IoU >= self.tri_thres:
            return self.tri_reward
        else:
            return -self.tri_reward

    def reset(self, isTrain = True):
        if isTrain:
            self.ground_truth, pic = self.data.get_data('train', self.sess)
        else:
            self.ground_truth, pic = self.data.get_data('test', self.sess)

        img = self.sess.run(pic)
        self.state = State(img, img.shape[0], img.shape[1])

        self._calc_IoU()

        return self.state

    def _sign(self, x):
        return 1 if x >= self.eps else -1
    def _isTerminal(self):
        if self.IoU >= self.accept_rate:
            return True
        else:
            return False

    def act(self, action):
        pre_IoU = self.IoU
        if action != 8:
            self._act(action)
        return self.state, self._sign(self.IoU - pre_IoU) if action != 8 else self.trigger_reward(), self._isTerminal() or action == 8

    def get_size(self, name):
        if name == 'train':
            return self.train_size
        else:
            return self.test_size
    def get_random_positive(self):
        possible = []
        for option in xrange(self.action_size):
            if option != 8:
                his_box = np.copy(self.state.box)
                his_IoU = self.IoU
                self._act(option)
                if self._sign(self.IoU - his_IoU) > 0:
                    possible.append(option)
                self.state.box = his_box
                self.IoU = his_IoU
            else:
                if self.trigger_reward() > 0:
                    possible.append(option)
        
        pos_cnt = len(possible)
        if pos_cnt != 0:
            return possible[np.random.randint(0, pos_cnt)]
        else:
            return np.random.randint(0, self.action_size)

    def end(self):
        self.coord.request_stop()
        self.coord.join(self.thread)
