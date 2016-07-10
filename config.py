class Config(object):
    def __init__(self):
        scale = 10000

        # Environment Settings
        self.mem_capacity = 23 * scale 
        self.action_size = 8 # Movement type of the bounding_box
        self.move_alpha = 0.2 # The movement size of the bounding_box 
        self.alpha = 0.2 # The rescale rate of the bounding_box
        self.eps = 1e-9
        self.isTrain = True

        # Network Settings
        self.ini_mean = 0.0
        self.ini_stddev = 0.02 # Ini_* are weights initialize hyperparameters
        self.bias_starter = 0.0 # Bias initialize hyperparameters
        self.learning_rate_minimum = 0.00025
        self.dqn_learning_rate = 0.015
        self.dqn_learning_rate_decay = 0.96
        self.dqn_learning_rate_decay_step = 5 * scale 
        self.dqn_momentum = 0.95
        self.dqn_epsilon = 0.01 # The epsilon hyperparameter of RMSPropOptimizer
        self.min_delta = -5 
        self.max_delta = 5 # The bound of delta
        
        # Training Settings
        self.epi_size = 50 # The episodes size
        self.step_size = 28
        self.check_point = 28  
        self.mx_to_keep = 15
        self.min_reward = -1 
        self.max_reward = 1
        self.act_ep = 0.8 # The epsilon hyperparameter of epsilon-policy
        self.batch_size = 5 
        self.learning_start_point = 30
        self.update_C = 8  
        self.discount = 0.75 
        self.accept_rate = 0.5
        
        # I/O Settings
        self.train_dir = "../VOCdevkit/VOC2012/JPEGImages/" 
        self.train_ano_dir = "../VOCdevkit/VOC2012/Annotations/"
        self.model_dir = "./Models/" # Snapshots directory
        self.test_dir = None
        self.test_ano_dir = None
