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
        self.count = 0
        self.mem_start = 0

    def add(self, sta, act, rew, nxt, term, his_code):
        if self.count >= self.mem_size:
            p = self.mem_start
            self.mem_start = (self.mem_start + 1) % self.mem_size
        else:
            p = self.mem_start + self.count
        self.count += 1
        
        self.s[p] = State(same = sta)
        self.act[p] = act
        self.rew[p] = rew
        self.nxt[p] = State(same = nxt)
        self.term[p] = term
        self.his_code[p] = his_code

    def sample(self, batch_size):    
        idx = np.random.randint(0, self.count, batch_size)
        idx = (idx + self.mem_start) % self.mem_size
        return self.s[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.term[idx], self.his_code[idx]
