#! /usr/bin/env python
"""
    File name: motion_visualization_tools.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: MIT
"""

##########################################################################################
# Imports
##########################################################################################
from src.main_utils import Configuration, ModelConfiguration
from src.motion_visualization_tools import test_scaling_method,test_generation_method, check_original_reconstructed_generated, compare_motion_data_plots
from src.cvae_networks import CVAE_CNN
from src.train_utils import train_model
from src.thirdHand_data_loader import get_min_max_from_dataset
import torch
import numpy as np

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

if __name__=="__main__":
    
    train_model = False
    model, project_config, model_config = create_the_model()

    if train_model:
        print("---------------------------------------------------------------------------------")
        print("Trainig the model from scratch, the model will be saved in ./models/{}.pt".format(model_config.model_name_to_save))
        train_model (model, project_config, model_config.epochs, model_config.model_name_to_save) 
    else:
        print("---------------------------------------------------------------------------------")
        print("Loading trained model from: ./models/{}.pt".format(model_config.model_name_to_save))
        try:
            model = torch.load("./models/{}.pt".format(model_config.model_name_to_save))
        except:
            print("A trained model does not exist in the provided path, please train the model first.")
            
            
    test_generation_method(model, project_config, generation_size= 16) 
    check_original_reconstructed_generated(model, project_config)