class Config(object):
    def __init__(self):
        scale = 10000

        # Environment Settings
        self.mem_capacity = 100 * scale 
        self.action_size = 8
        self.alpha = 0.5 
        self.eps = 1e-9
        self.isTrain = True

        # Network Settings
        self.ini_mean = 0.0
        self.ini_stddev = 0.02
        self.bias_starter = 0.0 
        self.learning_rate_minimum = 0.00025
        self.dqn_learning_rate = 0.0025
        self.dqn_learning_rate_decay = 0.96
        self.dqn_learning_rate_decay_step = 5 * scale 
        self.dqn_momentum = 0.95
        self.dqn_epsilon = 0.01
        self.min_delta = -1 
        self.max_delta = 1
        
        # Training Settings
        self.epi_size = 1000 
        self.step_size = 50
        self.check_point = 25  
        self.min_reward = -1 
        self.max_reward = 1
        self.act_ep = 0.4 
        self.batch_size = 5 
        self.learning_start_point = 30
        self.update_C = 8  
        self.discount = 0.75 
        self.pool_size = 4 * self.mem_capacity
        self.accept_rate = 0.65
        
        # I/O Settings
        self.train_dir = "../VOCdevkit/VOC2012/JPEGImages/" 
        self.train_ano_dir = "../VOCdevkit/VOC2012/Annotations/"
        self.test_dir = None
        self.test_ano_dir = None
