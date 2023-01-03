from src.thirdHand_data_loader import MotionSignalProcess, MezrabMotionDataset
from src.thirdHand_data_loader import read_motion_csv_files, test_data_dimensions, dataloader_creator

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import plotly.graph_objects as go

class Configuration(object):
    def __init__(self, device='cuda', 
                 csv_folder_path=None, 
                 tresh_l= 0.289, 
                 tresh_h_normal=0.4, 
                 tresh_h_riz=0.27, 
                 dist=15, 
                 peak_dist=30, 
                 motion_fixed_length=20, 
                 data_item="X_centered_scaled", 
                 batch_size=128):
        
        # general settings
        self.device = device
        
        # data processgin settings
        self.csv_folder_path = csv_folder_path   
        self.tresh_l = tresh_l 
        self.tresh_h_normal = tresh_h_normal
        self.tresh_h_riz = tresh_h_riz
        self.dist = dist
        self.peak_dist = peak_dist
        self.motion_fixed_length = motion_fixed_length
        
        # training setting
        self.batch_size = batch_size
        self.data_item = data_item
        self.epochs = 500
        self.writer = None
        
        if csv_folder_path is None:
            self.csv_folder_path = "./data_pipelines/motion_csv_files"
        else:
            self.csv_folder_path = csv_folder_path

        self.process_dataframe()
        self.process_dataset_dataloaders()
        
    def process_dataframe(self):
        strokes_df = read_motion_csv_files(self.csv_folder_path, [0,1,2,3,4,7,8,9,10,11,12]) 
        # reads the dataframe and converts it ontp slices of Mezrab strokes
        strokes_processor = MotionSignalProcess(strokes_df, 
                                                self.tresh_l, 
                                                self.tresh_h_normal, 
                                                self.dist, 
                                                self.peak_dist, 
                                                self.motion_fixed_length)

        # grabs all the final strokes from the object
        strokes = strokes_processor.strokes

        # process riz notes
        strokes_df_riz = read_motion_csv_files(self.csv_folder_path, [5,6]) 
        strokes_processor_riz = MotionSignalProcess(strokes_df_riz, 
                                                    self.tresh_l, 
                                                    self.tresh_h_riz, 
                                                    self.dist, 
                                                    self.peak_dist, 
                                                    self.motion_fixed_length)
        
        strokes_riz = strokes_processor_riz.strokes

        # stack the two set
        self.strokes = np.vstack([strokes, 
                                  strokes_riz])
        
    def process_dataset_dataloaders(self):
        # makes the dataoader out of the strokes
        self.mezrab_stroke_dataset = MezrabMotionDataset(self.strokes, 
                                                         self.device)
        data_pack = dataloader_creator(self.mezrab_stroke_dataset,
                                       batch_size=self.batch_size)
        self.train_dataset, self.valid_dataset, self.test_dataset, self.train_iterator, self.valid_iterator, self.test_iterator  = data_pack
    
    def init_writer(self):
        write_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        self.writer = SummaryWriter('runs/testTensorboard/test_{}'.format(write_name))