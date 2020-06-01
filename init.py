import torch 
import numpy as np
import pydicom


## Init function
class InitParser(object):
    def __init__(self):
        ## set gpu
        self.use_cuda = torch.cuda.is_available()
        self.gpu_id = 2
        self.version = "v1"
        self.num_workers = 20
    
        ## set parameters
        self.mode = "test"

        self.sparse_view = 60
        self.full_view = 1160

        # path setting
        if torch.cuda.is_available():
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo"
            self.root_path = "/mnt/tabgha/users/gy/MyProject/TradDa" 
        else:
            self.data_root_path = "V:/users/gy/data/Mayo"
            self.root_path = "V:/users/gy/MyProject/TradDa"

        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/" + self.version
        self.loss_path = self.result_path + "/loss"
        self.test_result_path = self.result_path + "/test_result"
        self.train_folder = {"patients": ["L096","L109","L143","L192","L286","L291","L310","L333", "L506"], "SliceThickness": ["full_3mm"]}
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
        self.val_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
