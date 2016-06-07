import tensorflow as tf

from dqn.agent import Agent
from config import Config

def main(args):

    tf.Session() as sess:
        config = Config() 
        player = Agent(config, sess)

    if flags.train:
        player.train()
    else:
        pass

if __name__ == '__main__':
    tf.app.run()
