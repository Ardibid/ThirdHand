#! /usr/bin/env python
"""
    File name: main_utils.py
    Author: Ardavan Bidgoli
    Date created: 11/03/2021
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: MIT
"""

import numpy as np
import plotly.graph_objects as go
import torch

def eval_epoch(
                model, 
                config, 
                plot= False, 
                scaled_plot= False, 
                update_tensorboard= False, 
                epoch=None, 
                loss_function= None, 
                show_one_sample= False,
                save_plot= False,
                path_to_save_plot = None,
                rec_loss= "L1", 
                reduction="sum", 
                kld_weight= 1e-1):
    """Evaluation function

    Args:
        model (VAE_CNN): the cvae model
        config (Configuration): represents the project configuration
        plot (bool, optional): Switch to turn plotting the results on of off. Defaults to False.
        scaled_plot (bool, optional): Defines if the plots should be scaled. Defaults to False.
        update_tensorboard (bool, optional):switch to update tensorboard. Defaults to False.
        epoch (int, optional): The epoch number of training. Defaults to None.
        loss_function (vae_loss_function, optional): VAE loss function. Defaults to None.
        show_one_sample (bool, optional): Switch to show one sample. Defaults to False.
        save_plot (bool, optional): Switch to save the plots. Defaults to False.
        path_to_save_plot (string, optional): path for saving eval plots. Defaults to None.
        rec_loss (str, optional): Reconstruction loss. Defaults to "L1".
        reduction (str, optional): loss reductino method. Defaults to "sum".
        kld_weight (float, optional): kld_loss to rec_loss ratio. Defaults to 1e-1.

    Returns:
        touple: batch_eval_loss/counter, batch_eval_rec_losses/counter, batch_eval_kld_losses/counter
    """
    # putting the model in eval mode
    model.eval()
    
    batch_eval_loss = 0
    batch_eval_rec_losses= 0 
    batch_eval_kld_losses= 0
    
    # iterating over the validation data iterator
    for i, data in enumerate(config.valid_iterator):
        # fetching x and y
        x= data[config.data_item]
        y= data["Y"]
        
        # passing the data through the model
        x_rec, mean, log_var = model(x,y)
        # calculating the loss
        eval_loss, eval_rec_losses, eval_kld_losses = loss_function(
                                                                    x, 
                                                                    x_rec, 
                                                                    log_var,
                                                                    mean, 
                                                                    rec_loss= rec_loss, 
                                                                    reduction= reduction, 
                                                                    kld_weight= kld_weight,
                                                                    )
        # updating the losses
        batch_eval_loss += eval_loss.item()
        batch_eval_rec_losses += eval_rec_losses.item()
        batch_eval_kld_losses += eval_kld_losses.item()
        
        # get the pot updated only for the first item in each batch
        if i == 0:
            if plot or update_tensorboard or save_plot:
                if scaled_plot:
                    if config.data_item == 2:
                        min = data[5][0]
                        max = data[6][0]
                        x = (x + min) * (max-min)
                        x_rec = (x_rec + min) * (max-min)

                    elif config.data_item == 4:
                        min = data[7][0]
                        max = data[8][0]
                        x = (x + min) * (max-min)
                        x_rec = (x_rec + min) * (max-min)               
                    
                fig = stroke_visualizer_mix(x, x_rec)
                
                if plot:
                    fig.show()
                
                if update_tensorboard:
                    fig.write_image(path_to_save_plot) 
                    image =  Image.open(path_to_save_plot)
                    image = np.asarray(image)
                    
                    config.writer.add_image(tag='test_progress', 
                                        img_tensor = image,
                                        global_step= epoch,
                                        dataformats='HWC') 
                if save_plot:
                    fig.write_image(path_to_save_plot) 
                    
    counter = len(config.valid_iterator)   
    
    
    return (batch_eval_loss/counter,
            batch_eval_rec_losses/counter, 
            batch_eval_kld_losses/counter)


def stroke_visualizer_mix(x, x_rec):
    """A function to show the samples on motions in a 3d plot

    Args:
        x (Tensor): dataset of motions, each motion should be of size 
        x_rec (Tensor): reconstruction of the motions. 
    """
    
    test_scatters = []
        
    plot_title ="Reconstruction of one sample"
    
    index = np.random.randint(0, x.shape[0])
    
    x = x[index,:,:]   
    x_rec = x_rec[index,:,:] 
    
    # adding the motion and motion reconstruction
    # more info here: https://plotly.com/python/3d-line-plots/
    x_plot = go.Scatter3d(x= x[:,0].detach().cpu().numpy(), 
                            y= x[: ,1].detach().cpu().numpy(), 
                            z= x[:, 2].detach().cpu().numpy(),  
                            mode='lines',
                            name="Motion",
                            line=dict(#color= "white", 
                                      width=3,),
                            )
    
    x_rec_plot = go.Scatter3d(x= x_rec[:,0].detach().cpu().numpy(), 
                                y= x_rec[: ,1].detach().cpu().numpy(), 
                                z= x_rec[:, 2].detach().cpu().numpy(),  
                                mode='lines',
                                name="Reconstruction",
                                line=dict(#color= "gray",
                                          width=3),
                            )
    test_scatters.append(x_plot)
    test_scatters.append(x_rec_plot)
    
    distance = torch.sum((x-x_rec)**2, axis= 1).detach().cpu().numpy()
    distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
      
    for frame in range(x.shape[0]):
        px = x[frame, :] 
        px_rec = x_rec[frame, :]
        
        data = torch.stack([px, px_rec])
        # more info here: https://plotly.com/python/3d-line-plots/
        error_plot = go.Scatter3d(x= data[:,0].detach().cpu().numpy(), 
                                y= data[: ,1].detach().cpu().numpy(), 
                                z= data[:, 2].detach().cpu().numpy(),  
                                mode='lines',
                                line=dict(
                                            color="rgba({}, {}, {}, 255)".format(distance[frame]*255,
                                                                                 50,
                                                                                 (1- distance[frame])*255),
                                            width=2,
                                        ),
                                showlegend=False,
                                )
        test_scatters.append(error_plot)
        
    
    
    fig = go.Figure(data=test_scatters)

    fig.update_layout(height=800,
                        margin=dict(l=5, r=5, t=50, b=50),
                        template =  "plotly_dark",
                        title_text= plot_title,
                        font=dict(family="Roboto, monospace",
                                size=12,
                                color="white"
                                ),
                        scene=dict(
                                    aspectratio = dict( x=1, y=1, z=1 ),
                                    camera=dict(up=dict(x=0, y=0, z=1),eye=dict(x=-1.5, y=1.5, z=1.5)),
                                ),
                        showlegend= True,
                        coloraxis_showscale=False,)

    fig.update_xaxes(showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        zerolinewidth=1, 
                        zerolinecolor='gray',
                        fixedrange= True,)

    fig.update_yaxes(showticklabels=False, 
                        showgrid=False, 
                        zeroline=False,
                        zerolinewidth=1, 
                        zerolinecolor='gray',
                        fixedrange= True,)
    
    return fig