import os
import inspect
import tensorflow as tf

from config import Config

def class_vars(obj):
    return {k:v for k, v in inspect.getmembers(obj) if not k.startswith('__') and not callable(k)}

class BaseModel(object):
    def __init__(self,config):
        self._saver = None
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
    
        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, step = None):
        print "[*] Now, saving checkpoints.........Path: " + self.model_dir
        model_name = type(self).__name__

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.saver.save(self.sess, self.model_dir + 'snapshot', global_step = step)

    def load_model(self):
        print "[*] Now, loading checkpoints..."

        chkp = tf.train.get_checkpoint_state(self.model_dir)
        if chkp and chkp.model_checkpoint_path:
            chkp_name = os.path.basename(chkp.model_checkpoint_path)
            fname = os.path.join(self.model_dir, chkp_name)
            self.saver.restore(self.sess, fname)
            print "[*] Now, load success: %s" % fname
            return True
        else:
            print "[!] Error! Load failed: %s" % self.model_dir
            return False

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep = self.config.mx_to_keep)
        return self._saver
