#! /usr/bin/env python
"""
    File name: cvae_networks.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: Attribution-NonCommercial-ShareAlike 4.0 International
"""

##########################################################################################
# Imports
##########################################################################################
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch import optim as optim

import numpy as np
from os.path import join
import time


##########################################################################################
# Encoder Network
##########################################################################################
class Encoder(nn.Module):
    def __init__(self, device, first_filter_size, kernel_size, depth, dropout, latent_dim):
        super(Encoder, self).__init__()
        self.device= device
        self.first_filter_size= first_filter_size
        self.kernel_size= kernel_size
        self.encoder_padding = kernel_size//2 -1
        self.depth = depth 
        self.latent_dim = 2**latent_dim
        self.filter_number = [2**(i) for i in range(first_filter_size+1)]
        self.filter_number.reverse()
        
        self.filter_number = self.filter_number[:self.depth]
        self.last_encoder_filter_size = None
        
        self.dropout = dropout
        self.encoder_layers = self.make_encoder() 
        
        self.last_filter_size = self.filter_number[0]
        self.last_feature_size= (10-(depth*2+1))
        self.last_dim =  self.last_filter_size*self.last_feature_size

        self.flatten_layer = nn.Flatten().to(device)
        self.convert_to_latent = nn.Linear(self.last_dim, 2*self.latent_dim).to(device)
        
    def make_encoder(self):
        encoder_cnn_blocks = []
        
        for i in range(len(self.filter_number)):
            if i ==0:
                in_dim = 20
                out_dim = self.filter_number[i]   
            else:
                in_dim = self.filter_number[i-1]
                out_dim = self.filter_number[i]
                
            cnn_block_layers=[
                            nn.Conv1d(in_channels= in_dim, 
                                    out_channels= out_dim, 
                                    kernel_size= self.kernel_size, 
                                    padding= self.encoder_padding),
                            nn.BatchNorm1d(out_dim),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            ]
            
            cnn_block = nn.Sequential(*cnn_block_layers).to(self.device)
            
            encoder_cnn_blocks.append(cnn_block)
            self.last_encoder_filter_size = out_dim
            
        self.filter_number.reverse()
        
        return nn.ModuleList(encoder_cnn_blocks)
    
    
    def reparametrization(self, mean, log_var):
        """
        Samples from a normal distribution with a given set of
        means and log_vars
        """
        # epsilon is a vector of size (1, latent_dim)
        # it is samples from a Standard Normal distribution
        # mean = 0. and std = 1.
        epsilon = torch.normal(mean= 0, std= 1, size = log_var.shape).to(self.device) 

        # we need to convert log(var) into var:
        var = torch.exp(log_var*0.5)
        # epsilon = torch.randn_like(var)
        # now, we change the standard normal distributions to
        # a set of non standard normal distributions
        z = mean + epsilon*var
        return z
    
    def forward(self, x, y):
        for block in self.encoder_layers:
            x = block(x) 

        latent_ready = self.flatten_layer(x) 
        latent = self.convert_to_latent(latent_ready)

        mean = latent[:, : self.latent_dim]
        log_var = latent[:,self.latent_dim:]

        z = self.reparametrization(mean, log_var)

        return z, mean, log_var
    

##########################################################################################
# Decoder Network
##########################################################################################
class Decoder(nn.Module):
    def __init__(self, device, first_filter_size, kernel_size, depth, latent_dim, last_filter_size, last_feature_size):
        super(Decoder, self).__init__()
        
        self.device= device
        self.first_filter_size= first_filter_size
        self.kernel_size= kernel_size
        self.encoder_padding = kernel_size//2 -1
        self.depth = depth 
        self.latent_dim = 2**latent_dim
        self.last_filter_size= last_filter_size
        self.last_feature_size= last_feature_size
                       
        self.filter_number = [2**(i) for i in range(first_filter_size+1)]        
        self.filter_number = self.filter_number[-self.depth:]
        self.filter_number.reverse()
        
        self.decoder_layers = self.make_decoder()
        self.z_to_decoder = nn.Linear(self.latent_dim,self.last_filter_size*self.last_feature_size).to(device)
       
    def make_decoder(self):
        decoder_cnn_blocks = []
        
        for i in range(len(self.filter_number)+1):
            self.decoder_padding = self.encoder_padding
            if i == 0:
                in_dim = self.last_filter_size
                out_dim = self.filter_number[i]
                
            elif i == len(self.filter_number):
                in_dim = self.filter_number[i-1]
                out_dim = 20  
                self.decoder_padding += 1
                
            else:
                in_dim = self.filter_number[i-1]
                out_dim = self.filter_number[i]
                        
            cnn_block_layers = [
                                nn.ConvTranspose1d(in_channels= in_dim, 
                                                    out_channels= out_dim, 
                                                    kernel_size= self.kernel_size, 
                                                    padding= self.decoder_padding,
                                                    ),
                                ]
            
            
            cnn_block = nn.Sequential(*cnn_block_layers).to(self.device)
            decoder_cnn_blocks.append(cnn_block)
            
        return nn.ModuleList(decoder_cnn_blocks)
    
    def forward(self, z, y):
        decoded = self.z_to_decoder(z).view(-1, self.last_filter_size, self.last_feature_size)
        
        for block in self.decoder_layers:
            decoded = block(decoded)
        
        return decoded     

    
##########################################################################################
# C-VAE Network
##########################################################################################    
class CVAE_CNN(nn.Module):
    def __init__(self, project_config, model_config):
        super(CVAE_CNN, self).__init__()
        
        self.project_config = project_config
        self.model_config = model_config
         
        self.encoder = Encoder(
                                self.project_config.device, 
                                self.model_config.first_filter_size, 
                                self.model_config.kernel_size, 
                                self.model_config.depth, 
                                self.model_config.dropout, 
                                self.model_config.latent_dim,
                                )

        self.decoder = Decoder(
                                self.project_config.device, 
                                self.model_config.first_filter_size, 
                                self.model_config.kernel_size, 
                                self.model_config.depth, 
                                self.model_config.latent_dim, 
                                self.encoder.last_filter_size,
                                self.encoder.last_feature_size,
                                )
        
        self.reduction = self.model_config.reduction
        self.kld_weight = self.model_config.kld_weight
        self.rec_loss = self.model_config.rec_loss
    
    def forward(self, x, y):
        z, mean, log_var = self.encoder(x, y)
        x_rec = self.decoder(z, y)
        return x_rec, mean, log_var
    
class VAE_CNN_(nn.Module):
    def __init__(self, device, first_filter_size, kernel_size, depth, dropout, latent_dim, rec_loss, reduction, kld_weight):
        super(VAE_CNN_, self).__init__()

        self.encoder = Encoder(
                                device, 
                                first_filter_size, 
                                kernel_size, 
                                depth, 
                                dropout, 
                                latent_dim,
                                )

        self.decoder = Decoder(
                                device, 
                                first_filter_size, 
                                kernel_size, 
                                depth, 
                                latent_dim, 
                                self.encoder.last_filter_size,
                                self.encoder.last_feature_size,
                                )
        
        self.reduction = reduction
        self.kld_weight = kld_weight
        self.rec_loss = rec_loss

    def forward(self, x, y):
        z, mean, log_var = self.encoder(x, y)
        x_rec = self.decoder(z, y)
        return x_rec, mean, log_var
        
##########################################################################################
# Support Function
##########################################################################################           
def vae_loss_function(x, x_rec, log_var, mean, rec_loss, reduction, kld_weight):
    """_summary_

    Args:
        x (torch.Tensor): batch of x
        x_rec (ntorch.Tensor): batch of x reconstructed after passing through the network
        log_var (torch.Tensorr): the logvar tensor from the encoder network
        mean (torch.Tensor):  the mean tensor from the encoder network
        rec_loss (string): loss function for reconstruction, i.e., "L1" or "L2"
        reduction (string): loss function reduction for reconstruction, i.e., "sum"
        kld_weight (float): rec_loss and kld_loss ration, i.e., 1e-1

    Returns:
        _type_: _description_
    """
    
    if rec_loss == "L1":
        train_rec_loss = F.l1_loss(x_rec, x, reduction=reduction)   
    else:
        train_rec_loss = F.mse_loss(x_rec, x, reduction=reduction)     
    train_kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp(), dim = 1), dim = 0)

    train_loss = train_rec_loss  + train_kld_loss*kld_weight
    
    return train_loss, train_rec_loss, train_kld_loss*kld_weight