import tensorflow as tf
import numpy as np

from .environtment import State

class Memory(object):
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.s = np.empty(mem_size, dtype = State)
        self.act = np.empty(mem_size, dtype = np.uint8)
        self.rew = np.empty(mem_size, dtype = int)
        self.nxt = np.empty(mem_size, dtype = State)
        self.term = np.empty(mem_size, dtype = bool)
        self.his_code = np.empty(mem_size, dtype = np.uint8)
        self.reset()

    def reset(self):
        self.full = False
        self.count = 0
        self.mem_start = 0
        self.mem_end = 0

    def add(self, sta, act, rew, nxt, term, his_code):
        if self.count == self.mem_size - 1:
            self.full = True
            self.count += 1
        if not self.full:
            self.count += 1
        
        p = self.mem_end
        self.s[p] = sta
        self.act[p] = act
        self.rew[p] = rew
        self.nxt[p] = nxt
        self.term[p] = term
        self.his_code[p] = his_code

        self.mem_end = (self.mem_end + 1) % self.mem_size
        if self.mem_end == self.mem_start:
            self.mem_start = (self.mem_start + 1) % self.mem_size

    def sample(self, batch_size):    
        idx = np.random.randint(0, self.count, batch_size)
        idx = (idx + self.mem_start) % self.mem_size
        return self.s[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.term[idx], self.his_code[idx]
