#! /usr/bin/env python
"""
    File name: motion_visualization_tools.py
    Author: Ardavan Bidgoli
    Date created: 01/03/2022
    Date last modified: 01/04/2022
    Python Version: 3.10.8
    License: MIT
"""

##########################################################################################
# Imports
##########################################################################################
import torch
import numpy as np
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

##########################################################################################
# Functions
##########################################################################################
def plot_layout_style(fig, 
                      plot_title, 
                      plot_height= 600,
                      theme= "plotly_dark"):
    """A function to unify the style of all plots

    Args:
        fig (plotly.graph_objs._figure.Figure): the graph to plot
        plot_title (string): title of the graph
        plot_height (int): height of the plot, default set to 600px
        theme (string): defines the plot thems, default set to "plotly_dark"
    Returns:
        _type_: _description_
    """
    fig.update_layout(height= plot_height,
                        margin=dict(l=5, r=5, t=50, b=50),
                        template =  theme,
                        title_text= plot_title,
                        font=dict(family="Roboto, monospace",
                                size=12,
                                color="white"
                                ),
                        scene=dict(
                                    aspectratio = dict( x=1, y=1, z=1 ),
                                    camera=dict(up=dict(x=0, y=0, z=1),
                                                eye=dict(x=-1.5, y=1.5, z=1.5)),
                                ),
                        showlegend= False,
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
                        fixedrange= True,
                        )
    return fig


def stroke_visualizer(dataset, 
                      samples_to_display= 64, 
                      scaled= False, 
                      centered = False,
                      plot_height= 600,
                      theme= "plotly_dark"):
    """ A function to show the samples on motions in a 3d plot
    Args:
        dataset (torch dataset): dataset of motions, each motion should be of size 
        samples_to_display (int, optional): number of samples to show in one plot. Defaults to 64.
        scaled (bool, optional): if the samples should be scaled to 1. Defaults to False.
        centered (bool, optional): if the samples should be centered aroun 0,0,0. Defaults to False.
        plot_height (int): height of the plot, default set to 800px
        theme (string): defines the plot thems, default set to "plotly_dark"
    """
    test_scatters = []
    plot_title = ""
    random_indices = np.random.choice(len(dataset), samples_to_display, replace=False)
    
    for i, d in enumerate(dataset):
        if i in random_indices:
            plot_title ="{} Motoins".format(samples_to_display)
            
            if scaled or centered:
                if scaled and centered: 
                    plot_title = "{}, Scaled, Centered".format(plot_title)
                    X = d[4]
                elif scaled:
                    plot_title = "{}, Scaled".format(plot_title)
                    X = d[2]
                else:
                    plot_title = "{}, Centered".format(plot_title)
                    X = d[3]
            else:
                X = d[0]
                
            color = np.arange(20)
            
            # more info here: https://plotly.com/python/3d-line-plots/
            tmp_plot = go.Scatter3d(x= X[:,0].detach().cpu().numpy(), 
                                    y= X[: ,1].detach().cpu().numpy(), 
                                    z= X[:, 2].detach().cpu().numpy(),  
                                    mode='lines',
                                    line=dict(
                                                color=color,
                                                width=2,
                                                colorscale= 'Agsunset', #'Agsunset', #'GnBu', 'Plasma', 'Sunset','Bluered_r'
                                        ),
                                    )
            test_scatters.append(tmp_plot)
            
    fig = go.Figure(data=test_scatters)

    fig.update_layout(height=plot_height,
                        margin=dict(l=5, r=5, t=50, b=5),
                        template =  theme,
                        title_text= plot_title,
                        font=dict(family="Roboto, monospace",
                                size=12,
                                color="white"
                                ),
                        scene=dict(
                                    aspectratio = dict( x=1, y=1, z=1 ),
                                    # camera=dict(up=dict(x=0, y=0, z=1),eye=dict(x=-1, y=-1, z=1)),
                                ),
                        showlegend= False,
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
    
    fig.show()
    
def show_generated_motions(x_generated, plot_title):
    all_plots = []
    color = np.arange(20)
    for index in range(x_generated.shape[0]):
        sample_generated = x_generated[index,:,:] 
        
        x_rec_plot = go.Scatter3d(
                                x= sample_generated[:,0].detach().cpu().numpy(), 
                                y= sample_generated[: ,1].detach().cpu().numpy(), 
                                z= sample_generated[:, 2].detach().cpu().numpy(),  
                                mode='lines',
                                name="Generated",
                                line=dict(color=color,
                                        width=2,
                                        colorscale= 'Agsunset',),
                                )
        all_plots.append(x_rec_plot)

    fig = go.Figure(data=all_plots)
    fig = plot_layout_style(fig, plot_title)
    return fig  

def show_generated_motions_advanced(motion_data, index = 0):
    all_plots = []
    colors= ['white', 'red', 'blue']
    names = ["Original", "Rec", "Gen"]
    for i, stroke_date in enumerate(motion_data):
        sample = stroke_date[index,:,:] 
        x_rec_plot = go.Scatter3d(x= sample[:,0].detach().cpu().numpy(), 
                                    y= sample[: ,1].detach().cpu().numpy(), 
                                    z= sample[:, 2].detach().cpu().numpy(),  
                                    mode='lines',
                                    name=names[i],
                                    line=dict(width=3, color= colors[i]),
                                    )
        all_plots.append(x_rec_plot)


    fig = go.Figure(data=all_plots)
    plot_title = "Generated Motion"
    
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
                        fixedrange= True,
                        scaleanchor = "x",
                        scaleratio = 1,)

    fig.show()
     
def compare_orig_rec_gen(model, project_config):
    sample_data= None
    sample_size = np.random.randint(0, project_config.batch_size)

    for d in project_config.train_iterator:
        x_samples = d[project_config.data_item][sample_size:sample_size+1, :,:]
        y_sampels = d['Y'][sample_size:sample_size+1, :,:] 

        z, __, __ = model.encoder(x_samples, y_sampels)

        noise = torch.normal(mean=.1, std=.1, size = z.shape).to(project_config.device)
        z = z + noise

        x_generated = model.decoder(z , y_sampels)
        x_rec,__, __ = model(x_samples, y_sampels)
        show_generated_motions_advanced([x_samples, x_rec,x_generated])
        break
    
def test_scaling_method(project_config):
    for d in project_config.train_iterator:
        original_centered_data = d["X_centered"]
        
        sample_centered_scaled = d[project_config.data_item]
        centered_max_val = d["centered_max_val"][0]
        centered_min_val = d["centered_min_val"][0]
        
        rescaled_data = (sample_centered_scaled*(centered_max_val-centered_min_val)+centered_min_val)
        compare_motion_data_plots([original_centered_data, rescaled_data], 0)
    
        break
    
def check_original_reconstructed_generated(model, project_config):
    sample_size = np.random.randint(0, 128, size= 16)
    fig = make_subplots(rows=1, cols=3)
    
    for d in project_config.train_iterator:
        x_samples = d[project_config.data_item][sample_size, :,:]
        y_sampels = d['Y'][sample_size, :,:] 
        
        z, __, __ = model.encoder(x_samples, y_sampels)

        noise = torch.normal(mean=.1, std=.21, size = z.shape).to(project_config.device)
        z = z + noise
    
        x_generated = model.decoder(z , y_sampels)
        x_rec,__, __ = model(x_samples, y_sampels)
        
        fig_1 = show_generated_motions(x_samples, "Original")
        fig_1.show()
        fig_2 = show_generated_motions(x_rec, "Reconstration")
        fig_2.show()
        fig_3 = show_generated_motions(x_generated, "Generated")
        fig_3.show()
        break
    
        
def compare_motion_data_plots(motion_data, index = 0):
    all_plots = []
    
    color = np.arange(20)
    color_scales = ['viridis', 'Agsunset']
    names = ["Original", "Generated"]
    widths= [4, 1]
    
    original_styel = dict(
                        color=color,
                        width= widths[0],
                        colorscale= color_scales[0],
                        )
                        
    generated_styel = dict(
                        color=color,
                        width=widths[1],
                        colorscale= color_scales[1],
                        )
    
    for i, stroke_date in enumerate(motion_data):
        sample = stroke_date[index,:,:] 
        line_style = generated_styel
        
        if i == 0:
            line_style = original_styel
        
        plot = go.Scatter3d(
                            x= sample[:,0].detach().cpu().numpy(), 
                            y= sample[: ,1].detach().cpu().numpy(), 
                            z= sample[:, 2].detach().cpu().numpy(),  
                            mode= 'lines',
                            # name= names[i],
                            line= line_style,
                            )
        all_plots.append(plot)


    scaler_val = 0.1
    plot = go.Scatter3d(
                            x= [-scaler_val, scaler_val], 
                            y= [-scaler_val, scaler_val], 
                            z= [0, scaler_val], 
                            mode="markers",
                            marker=dict(size= 0.1)
                        )
    all_plots.append(plot)
    fig = go.Figure(data=all_plots)
    plot_title = "Compare Motion Plots"
    fig = plot_layout_style(fig, plot_title)
    fig.show()
    return fig

def generation_range_visualization(motion_data):
    all_plots = []
    motion_cout = motion_data.shape[0]
    
    color = np.arange(20)
    color_scales = ['plotly3', 'teal', 'viridis', 'Agsunset']
    names = ["Original", "Generated"]
    widths= [4, 2]
    
    original_styel = dict(
                        color=color,
                        width= widths[0],
                        colorscale= color_scales[0],
                        )
                        
    generated_styel = dict(
                        color=color,
                        width=widths[1],
                        colorscale= color_scales[1],
                        )
    
    for i, sample in enumerate(motion_data):
        line_style = generated_styel
        if i <= motion_cout//2:
            shifter = (i-(motion_cout//2)-1)*0.05
        else:
            shifter = (i-(motion_cout//2))*0.05
        
        if i == 0:
            line_style = original_styel
            shifter= 0
            
        plot = go.Scatter3d(
                            x= sample[: ,0].detach().cpu().numpy()+shifter, 
                            y= sample[: ,1].detach().cpu().numpy(), 
                            z= sample[: ,2].detach().cpu().numpy(),  
                            mode= 'lines',# "lines+markers"
                            line= line_style,
                            )
        all_plots.append(plot)


    scaler_val = 0.1
    plot = go.Scatter3d(
                            x= [-scaler_val, scaler_val], 
                            y= [-scaler_val, scaler_val], 
                            z= [0, scaler_val], 
                            mode="markers",
                            marker=dict(size= 0.1)
                        )
    all_plots.append(plot)
    fig = go.Figure(data=all_plots)
    plot_title = "Series of Motion Generated by Navigating the Latent Space"
    fig = plot_layout_style(fig, plot_title)
    fig.show()
    return fig

def test_generation_method(model, project_config, generation_size=16):
    
    for j in range(1):
        sample_size = np.random.randint(0, 128, size=1)
        d_seed= np.random.randint(0, 10)
        for i, d in enumerate(project_config.train_iterator):
            if i ==d_seed:
                # picking a random sample
                x_samples = d[project_config.data_item][sample_size, :,:]
                y_sampels = d['Y'][sample_size, :,:] 
                                
                # getting the latent vector z
                z, __, __ = model.encoder(x_samples, y_sampels)

                # adding noise to latent vector
                noise = torch.normal(mean= .1, std= .2, size = (generation_size, 256)).to(project_config.device)
                z = z + noise
                
                # generating new motions based on the new z signals
                x_generated = model.decoder(z[:generation_size//2] , y_sampels[:generation_size//2])
                x_opposite_hand_generated = model.decoder(z[generation_size//2:] , y_sampels[generation_size//2:]*0) 
                
                # scaling to correct scale for the robot
                x_generated = scale_back(x_generated, d)
                
                x_opposite_hand_generated = scale_back(x_opposite_hand_generated, d)
                x_samples = scale_back(x_samples, d)     
            
                samples= torch.cat((x_samples, x_generated, x_opposite_hand_generated))   
                generation_range_visualization(samples)     
                 
                break
            
            
def quick_plot(train_losses=None, rec= None, KLD= None, eval_losses= None, log_scale= True):
    items_to_plot= []
    if train_losses:
        train = go.Scatter(x= np.arange(len(train_losses)), y=train_losses, name="training", mode='lines')
        items_to_plot.append(train)
    if eval_losses:
        valid = go.Scatter(x= np.arange(len(eval_losses)),  y=eval_losses, name="validation", mode='lines')
        items_to_plot.append(valid)
    
    if rec:
        rec = go.Scatter(x= np.arange(len(rec)), y=rec, name="reconstruction", mode='lines')
        items_to_plot.append(rec)
    
    if KLD:
        KLD = go.Scatter(x= np.arange(len(KLD)), y=KLD, name="KLD", mode='lines')
        items_to_plot.append(KLD)

    fig = go.Figure(items_to_plot)
    
    if log_scale:
        fig.update_yaxes(type="log")
        
    fig.update_layout(template =  "plotly_dark")
    fig.show()
    
def scale_back(motion, d):
    centered_max_val = d["centered_max_val"][0]
    centered_min_val = d["centered_min_val"][0]
    rescaled_data = (motion*(centered_max_val-centered_min_val)+centered_min_val)
    return rescaled_data