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
from src.main_utils import create_the_model
from src.motion_visualization_tools import test_scaling_method,test_generation_method, check_original_reconstructed_generated, compare_motion_data_plots
from src.train_utils import train_model
from src.thirdHand_data_loader import get_min_max_from_dataset
import torch
import numpy as np

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