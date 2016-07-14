class Config(object):
    def __init__(self):
        # Environment Settings
        self.mem_capacity = 19628 
        self.action_size = 9 # Movement type of the bounding_box
        self.trigger_reward = 3 # special reward for the trigger action
        self.trigger_threshold = 0.6 # reward threshold for the trigger action
        self.move_alpha = 0.15 # The movement size of the bounding_box 
        self.alpha = 0.15 # The rescale rate of the bounding_box
        self.eps = 1e-12
        self.isTrain = True 
        self.isLoadFromModel = False 
        self.act_his_len = 8 # The length of recent history

        # Network Settings
        self.ini_mean = 0.0
        self.ini_stddev = 0.02 # Ini_* are weights initialize hyperparameters
        self.bias_starter = 0.0 # Bias initialize hyperparameters
        self.learning_rate_minimum = 0.000025
        self.dqn_learning_rate = 0.0001
        self.dqn_learning_decay_rate = 0.96
        self.dqn_learning_decay_step = 2048
        self.min_delta = -1 
        self.max_delta = 1 # The bound of delta
        
        # Training Settings
        self.tot_epoches = 300 
        self.decay_epoches = 50 
        self.guide_epoches = 0 
        self.epi_size = 80000 # The episodes size
        self.step_size = 40 
        self.check_point = 100 
        self.mx_to_keep = 20 
        self.act_ep = 1. # The epsilon hyperparameter of epsilon-policy
        self.act_ep_threshold = 0.15 # The lower bound of epsilon
        self.batch_size = 1 
        self.update_C = 64 
        self.discount = 0.9 
        self.accept_rate = 0.95 
        self.train_start_point = 0 
        self.target_class = "person"
        self.dropout_prob = 0.3
        self.pretrained_model = "./Models/vgg16-20160129.tfmodel"

        # Testing Settings
        self.load_path = "./Snapshots/snapshot-1199"
        self.test_accept_rate = 0.5 
        
        # I/O Settings
        self.img_dir = "../VOCdevkit/VOC2012/JPEGImages/"
        self.ano_dir = "../VOCdevkit/VOC2012/Annotations/"
        self.train_list = "../VOCdevkit/VOC2012/ImageSets/Main/person_train.txt" 
        self.test_list = "../VOCdevkit/VOC2012/ImageSets/Main/person_val.txt"
        self.model_dir = "./Snapshots/" # Snapshots directory
