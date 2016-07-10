class Config(object):
    def __init__(self):
        # Environment Settings
        self.mem_capacity = 9628 
        self.action_size = 9 # Movement type of the bounding_box
        self.trigger_reward = 3 # special reward for the trigger action
        self.trigger_threshold = 0.5 # reward threshold for the trigger action
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
        self.dqn_learning_rate_decay_step = 250000 
        self.dqn_momentum = 0.95
        self.dqn_epsilon = 0.01 # The epsilon hyperparameter of RMSPropOptimizer
        self.min_delta = -5 
        self.max_delta = 5 # The bound of delta
        
        # Training Settings
        self.tot_epoches = 15
        self.epi_size = 150000 # The episodes size
        self.step_size = 40 
        self.check_point = 512  
        self.mx_to_keep = 15
        self.min_reward = -1 
        self.max_reward = 1
        self.act_ep = 0.8 # The epsilon hyperparameter of epsilon-policy
        self.ep_decay_inter = 512 # The time interval of epsilon decay
        self.ep_decay_step = 0.01 # The step of epsilon decay
        self.act_ep_threshold = 0.2 # The lower bound of epsilon
        self.batch_size = 8 
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
