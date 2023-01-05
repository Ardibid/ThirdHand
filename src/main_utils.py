#! /usr/bin/env python
"""
    File name: main_utils.py
    Author: Ardavan Bidgoli
    Date created: 11/03/2021
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: MIT
"""

##########################################################################################
# Imports
##########################################################################################
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.thirdHand_data_loader import MotionSignalProcess, MezrabMotionDataset
from src.thirdHand_data_loader import read_motion_csv_files, dataloader_creator
from src.cvae_networks import CVAE_CNN
##########################################################################################
# Classes
##########################################################################################
class Configuration(object):
    """A class to store all the project configuration (except C-VAE parameters)
    """
    def __init__(self, 
                 device='cuda', 
                 csv_folder_path=None, 
                 tresh_l= 0.289, 
                 tresh_h_normal= 0.4, 
                 tresh_h_riz= 0.27, 
                 dist= 15, 
                 peak_dist= 30, 
                 motion_fixed_length= 20, 
                 data_item="X_centered_scaled", 
                 batch_size= 128):
        """Initiates a Configuration object

        Args:
            device (str, optional): Selects CPU or GPU for CUDA. Defaults to 'cuda'.
            csv_folder_path (_type_, optional): Path to read the csv files of motion data. Defaults to None.
            tresh_l (float, optional): Determines how strong a lower peak should be in order to be recognized. Defaults to 0.289.
            tresh_h_normal (float, optional): Determines how strong an uppber peak should be in order to be 
                                              recognized for "Riz" motions segmentation. Defaults to 0.4.
            tresh_h_riz (float, optional): Determines how strong an uppber peak should be in order to be 
                                           recognized for not "Riz" motions segmentation. Defaults to 0.27.
            dist (int, optional): Minimum distance between each detected peak. Defaults to 15.
            peak_dist (int, optional): The max distance between the upper peak and lower peaks. Defaults to 30.
            motion_fixed_length (int, optional): size of each motion. Defaults to 20.
            data_item (str, optional): Determines which type of normalization and centering should be used. 
                                       Defaults to "X_centered_scaled".
            batch_size (int, optional): dataloader batch size. Defaults to 128.
        """
        
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
        # self.epochs = 500
        self.writer = None
        
        if csv_folder_path is None:
            self.csv_folder_path = "./data/motion_csv_files"
        else:
            self.csv_folder_path = csv_folder_path

        self.process_dataframe()
        self.process_dataset_dataloaders()
        
    def process_dataframe(self):
        """Reads the data csv files and converts them into a single numpy ndarray
        """
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
        self.strokes = np.vstack([strokes, strokes_riz])
        
    def process_dataset_dataloaders(self):
        """Creates a PyTorch dataoader out of the strokes
        """
        self.mezrab_stroke_dataset = MezrabMotionDataset(self.strokes, 
                                                         self.device)
        data_pack = dataloader_creator(self.mezrab_stroke_dataset,
                                       batch_size=self.batch_size)
        self.train_dataset, self.valid_dataset, self.test_dataset, self.train_iterator, self.valid_iterator, self.test_iterator  = data_pack
    
    def init_writer(self):
        """initiates the SummaryWriter for Tensorboard
        """
        write_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        self.writer = SummaryWriter('runs/testTensorboard/test_{}'.format(write_name))
        
        
class ModelConfiguration(object):
    """A class to store all the CVAE model configuration
    """
    def __init__(self, 
                 device='cuda', 
                 kernel_size = 5, 
                 first_filter_size = 9, 
                 depth = 2,
                 dropout = 0.1,
                 epochs = 100, 
                 latent_dim = 8,
                 rec_loss= "L1",
                 reduction= "sum",
                  kld_weight = 1e-1,
                 model_name_to_save="c_vae_model"
                 ):
        """_summary_
        Args:
            device (str, optional): Selects CPU or GPU for CUDA. Defaults to 'cuda'.
            kernel_size (int, optional): size of the kernel, both 3 and 5 would work. Defaults to 5.
            first_filter_size (int, optional): size of the first filter. Defaults to 9.
            depth (int, optional): depth of the encoder/decoder networks. Defaults to 2.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            epochs (int, optional): number of epochs to train the model. Defaults to 100.
            latent_dim (int, optional): size of the latent dim 2**latent_dim. Defaults to 8.
            rec_loss (string): loss function for reconstruction, i.e., "L1" or "L2"
            reduction (string): loss function reduction for reconstruction, i.e., "sum"
            kld_weight (float): rec_loss and kld_loss ration, i.e., 1e-1
            model_name_to_save (str, optional): name of the model to be saved. Defaults to "c_vae_model".
        """
        self.device= device 
        self.kernel_size = kernel_size
        self.first_filter_size = first_filter_size 
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.rec_loss= rec_loss
        self.reduction= reduction
        self.kld_weight = kld_weight
        self.model_name_to_save = model_name_to_save
    
def create_the_model(device='cuda', 
                     csv_folder_path=None, 
                     tresh_l= 0.289, 
                     tresh_h_normal= 0.4, 
                     tresh_h_riz= 0.27, 
                     dist= 15, 
                     peak_dist= 30, 
                     motion_fixed_length= 20, 
                     data_item="X_centered_scaled",
                     batch_size=128, 
                     kernel_size=5, # 3 or 5 both work!
                     first_filter_size =9, # between 10 and 5
                     depth = 2, # depth should be 2, 3, 4
                     dropout = 0.1,
                     epochs = 100, 
                     latent_dim = 8,
                     rec_loss= "L1",
                     reduction= "sum",
                     kld_weight = 1e-1,
                     model_name_to_save="c_vae_model"):
    """creates the configuration objects and inits the model
    For arguemtns documentations, refer to the src/main_utils.py
    Returns:
        touple: model, project_config, model_config
    """
    
    project_config = Configuration(device, 
                                   csv_folder_path,
                                   tresh_l, 
                                   tresh_h_normal, 
                                   tresh_h_riz, 
                                   dist, 
                                   peak_dist, 
                                   motion_fixed_length, 
                                   data_item, 
                                   batch_size)

    model_config = ModelConfiguration(
                                    project_config.device, 
                                    kernel_size, 
                                    first_filter_size, 
                                    depth,
                                    dropout,
                                    epochs, 
                                    latent_dim,
                                    rec_loss,
                                    reduction,
                                    kld_weight,
                                    model_name_to_save)
    
    model = CVAE_CNN(project_config, model_config)
    return model, project_config, model_config