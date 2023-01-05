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
import argparse



######################################################################
### Arguments
######################################################################
parser = argparse.ArgumentParser(prog='ThirdHand_trainer',
                                description="Trains a C-VAE for 6 DoF motion data")
parser.add_argument('-device', 
                    help="Defines which device to run the model on, cpu, or gpu (cuda)", 
                    default= "cuda",
                    nargs='?',
                    type=str)
parser.add_argument('-csv_folder_path', 
                    help="The folder that holds the motion cvs files", 
                    default= None,
                    nargs='?',
                    type=str)

parser.add_argument('-tresh_l', 
                    help="tresh_l", 
                    default= 0.289, 
                    nargs='?', 
                    type=float)

parser.add_argument('-tresh_h_normal', 
                    help="Determines how strong an uppber peak should be in order to be recognized"
                    ,default= 0.4, 
                    nargs='?', 
                    type=float)

parser.add_argument('-tresh_h_riz', 
                    help="Determines how strong an uppber peak should be in order to be recognized for Riz", 
                    default= 0.27, 
                    nargs='?', 
                    type=float)

parser.add_argument('-dist', 
                    help="Minimum distance between each detected peak", 
                    default= 15, 
                    nargs='?', 
                    type=int)

parser.add_argument('-peak_dist', 
                    help="The max distance between the upper peak and lower peak", 
                    default= 30, 
                    nargs='?', 
                    type=int)

parser.add_argument('-motion_fixed_length', 
                    help="The length of a given motion, from where the mezrab goes up, to the time it comes back up again", default= 30, nargs='?', type=int)


"""
data_frame (pandas dataframe): the data frame of all the mocap read from the csv files, 
tresh_l (float): Determines how strong a lower peak should be in order to be recognized  
                    higher values means it will ignore subtle touches, lower values mean it will considere 
                    any fluctuation as a low peak
tresh_h (float): Determines how strong an uppber peak should be in order to be recognized 
motion_fixed_length (int): The length of a given motion, from where the mezrab goes up, to the time it comes back up again!
"""

motion_fixed_length= 20
data_item="X_centered_scaled"
batch_size=128
kernel_size=5 # 3 or 5 both work!
first_filter_size =9 # between 10 and 5
depth = 2 # depth should be 2, 3, 4
dropout = 0.1
epochs = 100 
latent_dim = 8
rec_loss= "L1"
reduction= "sum",
kld_weight = 1e-1
model_name_to_save="c_vae_model"


args = parser.parse_args()
mode_selection = args.mode

device=args.device
csv_folder_path = args.csv_folder_path
tresh_l = args.tresh_l
tresh_h_normal = args.tresh_h_normal
tresh_h_riz = args.tresh_h_riz
dist = args.dist




















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