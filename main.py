import tensorflow as tf

from dqn.agent import Agent
from config import config_all

flags = tf.app.flags

def main(args):

    tf.Session() as sess:
        config = config_all(flags) 
        player = Agent(config, sess)

    if flags.train:
        player.train()
    else:
        pass

if __name__ == '__main__':
    tf.app.run()
