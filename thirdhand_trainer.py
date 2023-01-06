#! /usr/bin/env python
"""
    File name: motion_visualization_tools.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/06/2022
    Python Version: 3.10.8
    License: Attribution-NonCommercial-ShareAlike 4.0 International
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
from os import path

######################################################################
### Arguments
######################################################################
parser = argparse.ArgumentParser(prog = 'ThirdHand_trainer',
                                description = "Trains a C-VAE for 6 DoF motion data")

parser.add_argument('-mode', 
                    help = "Trainig the model (train) or loading a trained model (load)", 
                    default = "train",
                    nargs = '?',
                    type = str)


parser.add_argument('-device', 
                    help = "Defines which device to run the model on, cpu, or gpu (cuda)", 
                    default = "cuda",
                    nargs = '?',
                    type = str)
parser.add_argument('-csv_folder_path', 
                    help = "The folder that holds the motion cvs files", 
                    default = None,
                    nargs = '?',
                    type = str)

parser.add_argument('-tresh_l', 
                    help = "tresh_l", 
                    default = 0.289, 
                    nargs = '?', 
                    type = float)

parser.add_argument('-tresh_h_normal', 
                    help = "Determines how strong an uppber peak should be in order to be recognized"
                    ,default = 0.4, 
                    nargs = '?', 
                    type = float)

parser.add_argument('-tresh_h_riz', 
                    help = "Determines how strong an uppber peak should be in order to be recognized for Riz", 
                    default = 0.27, 
                    nargs = '?', 
                    type = float)

parser.add_argument('-dist', 
                    help = "Minimum distance between each detected peak", 
                    default = 15, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-peak_dist', 
                    help = "The max distance between the upper peak and lower peak", 
                    default = 30, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-motion_fixed_length', 
                    help = "The length of a given motion, from where the mezrab goes up, to the time it comes back up again", 
                    default = 20, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-data_item', 
                    help = "which scaling and centering to work with", 
                    default = "X_centered_scaled", 
                    nargs = '?', 
                    type = str)

parser.add_argument('-batch_size', 
                    help = "dataloader batch size", 
                    default = 128, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-kernel_size', 
                    help = "CNN Kernel size, 3 or 5 both work", 
                    default = 5, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-first_filter_size', 
                    help = "number of filters in the first layer as 2**n while 5<n<10, ", 
                    default = 9, 
                    nargs = '?',
                    type = int)

parser.add_argument('-depth', 
                    help = "Depth of the encoder, can be 2,3,4", 
                    default = 2, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-dropout', 
                    help = "dropout rate, keep it close to .1", 
                    default = 0.1, 
                    nargs = '?', 
                    type = float)

parser.add_argument('-epochs', 
                    help = "numebr of epochs to train the model, 150 is sweet", 
                    default = 150, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-latent_dim', 
                    help = "Size of the latent dimension as 2**n", 
                    default = 8, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-rec_loss', 
                    help = "reconstruction loss function, can be L1 or L2", 
                    default = 30, 
                    nargs = '?', 
                    type = int)

parser.add_argument('-reduction', 
                    help = "loss function reduction method, sum or mean", 
                    default = "sum", 
                    nargs = '?', 
                    type = str)

parser.add_argument('-kld_weight', 
                    help = "ratio of rec loss to kld loss in the loss function", 
                    default = 1e-1, 
                    nargs = '?', 
                    type = float)

parser.add_argument('-model_name_to_save', 
                    help = "name used to save the model in ./models", 
                    default = "c_vae_model", 
                    nargs = '?', 
                    type = str)



if __name__=="__main__":
    args = parser.parse_args()
    model, project_config, model_config = create_the_model(device = args.device, 
                                                            csv_folder_path = args.csv_folder_path, 
                                                            tresh_l = args.tresh_l, 
                                                            tresh_h_normal = args.tresh_h_normal, 
                                                            tresh_h_riz = args.tresh_h_riz, 
                                                            dist = args.dist, 
                                                            peak_dist = args.peak_dist, 
                                                            motion_fixed_length = args.motion_fixed_length, 
                                                            data_item = args.data_item,
                                                            batch_size = args.batch_size, 
                                                            kernel_size = args.kernel_size,  
                                                            first_filter_size = args.first_filter_size, 
                                                            depth = args.depth, 
                                                            dropout = args.dropout,
                                                            epochs = args.epochs, 
                                                            latent_dim = args.latent_dim,
                                                            rec_loss = args.rec_loss,
                                                            reduction = args.reduction,
                                                            kld_weight = args.kld_weight,
                                                            model_name_to_save = args.model_name_to_save)

    if args.mode == "train":
        print("---------------------------------------------------------------------------------")
        print("Trainig the model from scratch, the model will be saved in ./models/{}.pt".format(model_config.model_name_to_save))
        train_model (model, project_config, model_config, model_config.model_name_to_save) 
    
    else:
        print("---------------------------------------------------------------------------------")
        print("Loading trained model from: ./models/{}.pt".format(model_config.model_name_to_save))
        
        path_to_model = "./models/{}.pt".format(model_config.model_name_to_save)
        
        if path.exists(path_to_model):
            try:
                model = torch.load(path_to_model)
            except:
                print("Could not load the model.")
        else:
            print("A trained model does not exist in the provided path, please train the model first.")    
            
    test_generation_method(model, project_config, generation_size= 16) 
    check_original_reconstructed_generated(model, project_config)