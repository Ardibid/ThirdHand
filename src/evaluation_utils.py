import numpy as np
import plotly.graph_objects as go
import torch


def eval_epoch(
                model, 
                config, 
                data_iterator, 
                plot= False, 
                scaled_plot= False, 
                update_tensorboard= False, 
                epoch=None, 
                loss_function= None, 
                show_one_sample= False,
                save_plot= False,
                path_to_save_plot = None,
                is_vae= False,
                rec_loss= "L1", 
                reduction="sum", 
                kld_weight= 1e-1):
    
    
    model.eval()
    batch_eval_loss = 0
    batch_eval_rec_losses= 0 
    batch_eval_kld_losses= 0
    
    item_to_show = np.random.randint(len(config.valid_iterator))
    
    for i, data in enumerate(config.valid_iterator):
        x= data[config.data_item]
        y= data["Y"]
        
        x_rec = model(x, y)
        if is_vae:
            x_rec, mean, log_var = model(x,y)
            eval_loss, eval_rec_losses, eval_kld_losses = loss_function(
                                                                        x, 
                                                                        x_rec, 
                                                                        log_var,
                                                                        mean, 
                                                                        rec_loss= "L1", 
                                                                        reduction='sum', 
                                                                        kld_weight= kld_weight,
                                                                        )
            batch_eval_loss += eval_loss.item()
            batch_eval_rec_losses += eval_rec_losses.item()
            batch_eval_kld_losses += eval_kld_losses.item()
        
        else:
            eval_loss = loss_function(x, x_rec)
            batch_eval_loss += eval_loss.item()
        
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
    
    if is_vae:
        return (batch_eval_loss/counter,
                batch_eval_rec_losses/counter, 
                batch_eval_kld_losses/counter)
    else:     
        return batch_eval_loss/counter               


def stroke_visualizer_mix(x, x_rec):
    """A function to show the samples on motions in a 3d plot

    Args:
        dataset (torch dataset): dataset of motions, each motion should be of size 
        samples_to_display (int, optional): [description]. Defaults to 64.
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