#! /usr/bin/env python
"""
    File name: cvae_networks.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: Attribution-NonCommercial-ShareAlike 4.0 International
"""

import torch
from torch import optim as optim

import time

from .cvae_networks import vae_loss_function
from .evaluation_utils import eval_epoch
from .motion_visualization_tools import quick_plot

##########################################################################################
# Training Functions
##########################################################################################
def train_epoch_single(model, project_config, optimizer):
    """Function to perform a single epoch training
    Args:
        model (VAE_CNN): the cvae model
        project_config (Configuration): represents the project configuration
        optimizer (torch.optim): optimizer 

    Returns:
        array: a list of len 3, containing the average epoch_loss, epoch_rec_loss, epoch_kld_loss
    """
    # set the model in training mode
    model.train()
    
    epoch_loss = 0
    epoch_rec_loss = 0
    epoch_kld_loss = 0
 
    # iterate over the training data
    for data in project_config.train_iterator:
        x= data[project_config.data_item]
        y= data["Y"]
        
        optimizer.zero_grad()
        x_rec, mean, log_var = model(x,y)
        
        # calculating losses
        train_loss, train_rec_loss, train_kld_loss = vae_loss_function (x, 
                                                                        x_rec, 
                                                                        log_var, 
                                                                        mean, 
                                                                        rec_loss= model.rec_loss, 
                                                                        reduction= model.reduction,
                                                                        kld_weight= model.kld_weight)

        # updating the history
        epoch_rec_loss += train_rec_loss.item()
        epoch_kld_loss += train_kld_loss.item()
        epoch_loss += train_loss.item() 
        
        train_loss.backward()
        optimizer.step()
 
    counter = len(project_config.train_iterator)   
    results = [epoch_loss/counter,
                epoch_rec_loss/counter, 
                epoch_kld_loss/counter]      
    
    return results

def train_model(model, project_config, model_config= None, model_name_to_save="cvae_model", report_interval =50):  
    """Trains the cvae model for given number of epochs

    Args:
        model (VAE_CNN): the cvae model to train
        project_config (Configuration): represents the project configuration
        model_config (int, optional): number of training epochs. Defaults to 100.
        model_name_to_save (str, optional): name for the trained model to save. 
                                            Defaults to "cvae_model".

    Returns:
        touple: four lsits of data, train_losses ,train_rec_losses,train_kld_losses, eval_losses 
    """
    if model_config:
        epochs = model_config.epochs
    else:
        epochs = 100
        
    # setting up the trainnig process
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    
    project_config.process_dataset_dataloaders()
    
    train_losses = []
    train_rec_losses = []
    train_kld_losses = []
    eval_losses = []
    
    # main training loop
    for epoch in range(epochs+1):
        # train for a single epoch
        train_loss, train_rec_loss, train_kld_loss = train_epoch_single(
                                                                        model, 
                                                                        project_config,
                                                                        optimizer,
                                                                        )
        log_plot= False
                
        # interval report of model's reconstruction as a plot 
        if epoch % report_interval ==0:
            log_plot= True
            path_to_save_plot = "runs/progress/tmp_fig_{}.png".format(epoch)  
            print("Image {} saved".format(epoch)) 
        
        # calculating evaluation loss     
        eval_loss,_,_ = eval_epoch(
                                    model, 
                                    project_config, 
                                    loss_function= vae_loss_function,
                                    rec_loss= model.rec_loss, 
                                    reduction= model.reduction, 
                                    kld_weight= model.kld_weight,
                                    save_plot= log_plot,
                                    path_to_save_plot= path_to_save_plot,
                                    )
        
        # storing the results 
        train_losses.append(train_loss)
        train_rec_losses.append(train_rec_loss)
        train_kld_losses.append(train_kld_loss)
        eval_losses.append(eval_loss)
        
        
        if epoch % report_interval ==0:
            print("{}:\tTotal: {:.5f}\tEval loss: {:.5f}\t Rec loss: {:.5f}\t KLD loss: {:.5f}\t time: {:.1f}s".format(epoch, 
                                                                          train_loss,
                                                                          eval_loss,
                                                                          train_rec_loss,
                                                                          train_kld_loss,
                                                                          time.time()-start_time))
    
    # final report and saving the model
    quick_plot(train_losses=train_losses, eval_losses=eval_losses, KLD=None, rec=None)
    #plot_reports(train_losses, train_rec_losses, train_kld_losses, eval_losses)
    torch.save(model, "./models/{}.pt".format(model_name_to_save))
    
    return train_losses ,train_rec_losses,train_kld_losses, eval_losses


def plot_reports(train_losses, train_rec_losses, train_kld_losses, eval_losses):
    """A simple function to make 3 plots of the training process
       Since the kld loss is smaller than the rec loss, keeping all in the 
       same plots renders it hard to read, so each plot behaves slightly differently

    Args:
        train_losses (list): train loss
        train_rec_losses (list): train reconstratction loss
        train_kld_losses (list): train KLD loss
        eval_losses (list): evaluation loss
    """
    quick_plot(train_losses, train_rec_losses, train_kld_losses, eval_losses)
    quick_plot(train_losses=train_losses, eval_losses=eval_losses, KLD=None, rec=None)
    quick_plot(train_losses=train_losses, eval_losses=eval_losses, KLD=train_kld_losses, rec= train_rec_losses)