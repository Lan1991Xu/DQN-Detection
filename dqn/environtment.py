import tensorflow as tf
import numpy as np
import os

from dataset import Dataset
from memory import Memory

class Env(object):
    def __init__(self, config):
        self._reset(config)

    def _reset(self, config = None):

    def reset(self):
        self.reset_reset(config)

