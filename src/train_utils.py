#! /usr/bin/env python
"""
    File name: cvae_networks.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: MIT
"""
import torch
from torch import optim as optim

import time

from .cvae_networks import vae_loss_function
from .evaluation_utils import eval_epoch
from .motion_visualization_tools import quick_plot



def train_epoch_single(model, model_config, optimizer):
    model.train()
    epoch_loss = 0
    epoch_rec_loss = 0
    epoch_kld_loss = 0
 
    for data in model_config.train_iterator:
        x= data[model_config.data_item]
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
 
    counter = len(model_config.train_iterator)   
    results = [epoch_loss/counter,
                epoch_rec_loss/counter, 
                epoch_kld_loss/counter]      
    
    return results

def train_model(model, model_config, epochs=100, model_name_to_save="cvae_model"):         
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    
    
    model_config.process_dataset_dataloaders()
    train_losses = []
    train_rec_losses = []
    train_kld_losses = []
    eval_losses = []
    for epoch in range(epochs+1):
        # train
        train_loss, train_rec_loss, train_kld_loss = train_epoch_single(
                                                                        model, 
                                                                        model_config,
                                                                        optimizer,
                                                                        )
        log_plot= False
        report_interval = 50
        if epoch % report_interval ==0:
            log_plot= True
            path_to_save_plot = "runs/progress/tmp_fig_{}.png".format(epoch)  
            print("<<Image {} saved>>".format(epoch)) 
              
        eval_loss, _, _ = eval_epoch(
                                    model, 
                                    model_config, 
                                    model_config.valid_iterator, 
                                    loss_function= vae_loss_function,
                                    rec_loss= model.rec_loss, 
                                    reduction=model.reduction, 
                                    kld_weight= model.kld_weight,
                                    save_plot= log_plot,
                                    path_to_save_plot= path_to_save_plot,
                                    is_vae= True
                                    )
        
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
    quick_plot(train_losses, train_rec_losses, train_kld_losses, eval_losses)
    quick_plot(train_losses=train_losses, eval_losses=eval_losses, KLD=None, rec=None)
    quick_plot(train_losses=train_losses, eval_losses=eval_losses, KLD=train_kld_losses, rec= train_rec_losses)