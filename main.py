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
from src.motion_visualization_tools import test_scaling_method, test_generation_method
from src.cvae_networks import CVAE_CNN
from src.train_utils import train_model


def create_the_model():
    """creates the configuration objects and inits the model

    Returns:
        _type_: _description_
    """
    project_config = Configuration(batch_size = 128)
    model_config = ModelConfiguration(device='cuda', 
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
                        )
    model = CVAE_CNN(project_config, model_config)
    return model, project_config, model_config

def test_results(model, project_config):
    """Some visualizations for checking results

    Args:
        model (CVAE_CNN): the C-VAE model
        project_config (Configuration): project configuration data
    """
    test_scaling_method(project_config)    
    test_generation_method(model, project_config) 
         
if __name__=="__main__":
    model, project_config, model_config = create_the_model()
    train_model (model, project_config, model_config.epochs, model_config.model_name_to_save) 
    test_results(model, project_config)
    
       