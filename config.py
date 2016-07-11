class Config(object):
    def __init__(self):
        # Environment Settings
        self.mem_capacity = 628 
        self.action_size = 9 # Movement type of the bounding_box
        self.trigger_reward = 3 # special reward for the trigger action
        self.trigger_threshold = 0.6 # reward threshold for the trigger action
        self.move_alpha = 0.2 # The movement size of the bounding_box 
        self.alpha = 0.2 # The rescale rate of the bounding_box
        self.eps = 1e-12
        self.isTrain = True 
        self.isLoadFromModel = True 
        self.act_his_len = 8 # The length of recent history

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
        self.decay_epoches = 5
        self.epi_size = 80000 # The episodes size
        self.step_size = 40 
        self.check_point = 64  
        self.mx_to_keep = 15
        self.min_reward = -1 
        self.max_reward = 1
        self.act_ep = 0.8 # The epsilon hyperparameter of epsilon-policy
        self.act_ep_threshold = 0.15 # The lower bound of epsilon
        self.batch_size = 8 
        self.learning_start_point = 30
        self.update_C = 16  
        self.discount = 0.75 
        self.accept_rate = 0.95 
        self.train_start_point = 1536

        # Testing Settings
        self.load_path = "./Models/snapshot-1536"
        self.test_accept_rate = 0.5 
        
        # I/O Settings
        self.img_dir = "../VOCdevkit/VOC2012/JPEGImages/"
        self.ano_dir = "../VOCdevkit/VOC2012/Annotations/"
        self.train_list = "../VOCdevkit/VOC2012/ImageSets/Main/person_train.txt" 
        self.test_list = "../VOCdevkit/VOC2012/ImageSets/Main/person_val.txt"
        self.model_dir = "./Models/" # Snapshots directory
