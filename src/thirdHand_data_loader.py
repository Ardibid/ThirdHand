'''
    File name: thirdhand_data_loader.py
    Author: Ardavan Bidgoli
    Date created: 10/27/2021
    Date last modified: 01/03/2022
    Python Version: 3.10.8
    License: Attribution-NonCommercial-ShareAlike 4.0 International
'''

##########################################################################################
# Imports
##########################################################################################
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
import torch

from openTSNE import TSNE
import peakutils

import plotly.graph_objects as go
import plotly.express as px

import os


##########################################################################################
# Helper Functions
##########################################################################################
def read_motion_csv_files(path, files_to_read_indx= None):
    """Scans a given folder, finds all the files with specific name and extension (csv I mean!)
    Then concat them in one big dataframe

    Args:
        path (string): path to the folder that contains the data

    Returns:
        pandas data frame: all the data from the csv files, organized as a pandas dataframe
    """

    motion_csv_files =  os.listdir(path)
    motion_csv_files = [f for f in motion_csv_files if f[-14:]== "_processed.csv"]
    
    if files_to_read_indx is not None:
        motion_csv_files = [motion_csv_files[i] for i in files_to_read_indx]

    df = None
    df_sizes = []
    
    for f in motion_csv_files:
        f_path = os.path.join(path, f)
        data_tmp = pd.read_csv(f_path,  )
        
        if df is None:
            df = data_tmp
            
        else:
            df = pd.concat([df, data_tmp], ignore_index=True)
        df_sizes.append(df.shape[0])
                 
    print ("Data loaded from {} filse, stored in a dataframe with shape {}".format(len(motion_csv_files), df.shape))
    print ("Dataframe headers are: {}".format(list(df.columns)))
    
    return df

def get_min_max_from_dataset(samples, verbose= False):
    """
    Collects a dataset, and then return the min and max values over the last dimensions
    Args:
        samples (pytroch tensor): the data taht we want to find its min and max, if the dim of the dataset is 100, 20, 9,
                                        the result is of shape 9

    Returns:
        [pytorch tensor]: min and max tensors, of shape 9, as a list 
    """
    # min and max on the last dim
    min_val,_ = samples.min(axis=0)
    min_val,_ = min_val.min(axis=0)
    max_val,_ = samples.max(axis=0)
    max_val,_ = max_val.max(axis=0)
    
    # quick report
    if verbose:
        print ("Minimum values over the 9 values are: \n{}".format(min_val))
        print ("Maximum values over the 9 values are: \n{}".format(max_val))

    return min_val, max_val

def dataloader_creator(raw_dataset, training_ratio= 0.7, validation_ratio= .15, test_ratio= 0.15, batch_size = 16):
    """
    Converts the raw data into test/train dataloaders and iterators

    Args:
        raw_dataset (pytroch util dataset): the main dataset to split
        training_ratio (float, optional): ratio of training data to all available samples Defaults to 0.7.
        validation_ratio (float, optional): ratio of validation data to all available samples Defaults to .15.
        test_ratio (float, optional): ratio of test data to all available samples. Defaults to 0.15.
        batch_size (int, optional): size of each batch. Defaults to 16.

    Returns:
        [type]: [description]
    """
    assert (training_ratio + validation_ratio + test_ratio == 1)
    
    training_size = int(raw_dataset.X.shape[0]*(training_ratio))
    validation_size = int(raw_dataset.X.shape[0]*(validation_ratio))
    test_size = raw_dataset.X.shape[0] - training_size - validation_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(raw_dataset,[training_size,
                                                                                            validation_size,
                                                                                            test_size])
    
    train_iterator = DataLoader(train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    drop_last= True)
    
    valid_iterator = DataLoader(valid_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    drop_last= True)
    
    test_iterator = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    drop_last= True)
   
    
    return train_dataset, valid_dataset, test_dataset, train_iterator, valid_iterator, test_iterator
    
##########################################################################################
# Motion pre-processing class
##########################################################################################
class MotionSignalProcess(object):
    def __init__(self, data_frame, tresh_l, tresh_h, dist, peak_dist, motion_fixed_length):
        """A class of tools to process the data from the motion capture system to get it ready for ML,
        A NOTE TO MYSELF: the default tresh_l and tresh_h are suitable for the casual motions, it will ignore RIZ ones

        Args:
            data_frame (pandas dataframe): the data frame of all the mocap read from the csv files, 
            tresh_l (float): Determines how strong a lower peak should be in order to be recognized  
                             higher values means it will ignore subtle touches, lower values mean it will considere 
                             any fluctuation as a low peak
            tresh_h (float): Determines how strong an uppber peak should be in order to be recognized 
            dist (int):  Minimum distance between each detected peak 
            peak_dist (int): The max distance between the upper peak and lower peak
            motion_fixed_length (int): The length of a given motion, from where the mezrab goes up, to the time it comes back up again!
        """
        self.data_frame = data_frame
        self.signal_z = self.data_frame["pz"] 
        self.signal_y = self.data_frame["py"] 
        
        self.down_idxs, self.high_idxs = self.find_all_peaks(self.signal_z, tresh_l, tresh_h, dist)
        self.down_idxs = self.filter_peaks(self.signal_z, self.signal_y, self.down_idxs, self.high_idxs, peak_dist)
        self.strokes = self.slice_motion_sequence(self.data_frame, motion_fixed_length, self.down_idxs)


    def find_all_peaks(self, signal, tresh_l, tresh_h, dist):
        """
        Reads the signal, a nx1 numpy array and finds the index values for all peaks
        Args:
            signal (numpy array): nx1 numpy array
            tresh_l (float): Determines how strong a lower peak should be in order to be recognized 
            tresh_h (float): Determines how strong an uppber peak should be in order to be recognized
            dist (int): Minimum distance between each detected peak.
        Returns:
            down_idxs (list of ints): indices of local minimums
            high_idxs (list of ints): indices of local maximums
        """

        down_idxs = peakutils.indexes(-1*signal, thres=tresh_l, min_dist=dist)
        high_idxs = peakutils.indexes(signal, thres=tresh_h, min_dist=dist)
        return down_idxs, high_idxs


    def filter_peaks(self, signal, signal_y, down_idxs, high_idxs, peak_dist):
        """
        Filters the false peaks, and only returns the ones that are the real 
        tourching points
        Arguments:
            signal (np.array) 1xN: the z value of all motions in as a super long stream (17391,)
            signal_y (np.array) 1xN: the y value of all motions in as a super long stream (17391,) 
            down_idxs (np.array) 1xN: the indices of all recognized lower peaks 
            high_idxs (np.array) 1xN: the indices of all recognized upper peaks 
            peak_dist (int): the max distance between the upper peak and lower peak
        """

        # finds the nearest low peak to a high peak (we are certain that high peaks are always followed by a touch)
        closest_idx = np.searchsorted(down_idxs, high_idxs, side='left')
        
        # makes sure that the index is not out of range of the down_index list
        closest_idx = closest_idx[np.where(closest_idx< down_idxs.shape[0])]

        # ok, these are the touch points!
        touch_frames_idx = np.array(down_idxs[closest_idx])

        # we filter the touch points that are too far from the high peaks, 
        # too far means the peak is actually just a rest position not the acceleration point
        distance_filter = np.where(touch_frames_idx - high_idxs[:touch_frames_idx.shape[0]] < peak_dist)
        filtered_by_dist = touch_frames_idx[distance_filter[0]]
        results = filtered_by_dist
        

        # here we filter all local minimums that are actually too high for being a real
        # touch point! If this is a touch point, then it should be close to strings,
        # not somewhere at the middle of the air
        min_val = np.min(signal) 
        max_val= np.max(signal)
        
        # only passes the peaks that are %10 higher than the lowest points
        percentage_filter = np.where((signal[filtered_by_dist]-min_val)/(max_val-min_val) < .20)
        filtered_by_percentage= filtered_by_dist[percentage_filter]
        
        # remove all the motions with very high change in y near touching point
        results = []

        for i in filtered_by_percentage:
            y_vals = signal_y[i-5:i+5]
            z_vals = signal[i-5:i+5]
            y_range = np.max(y_vals)-np.min(y_vals)
            z_range = np.max(z_vals)-np.min(z_vals)
            
            if y_range < z_range:
                results.append(i)
                
        results = np.array(results)  

        return results

    def slice_motion_sequence(self, data_frame, motion_fixed_length, down_idxs):
        """
        Grabs the motions and cut them in equal lenghtes with the touching point at the middle
        Args:
            data_frame (pandas dataframe): the dataframe made by all the motion from the csv files
            motion_fixed_length (int): size of each motion, i.e., 20
            down_idxs (list of int): index of real touching motions

        Returns:
            [np.array]: motions as seperated sequence of data with shape of motion_fixed_length x 9
        """
        span = motion_fixed_length//2
        strokes = None

        for d in down_idxs:
            stroke = data_frame.loc[d-span+1:d+span].to_numpy()
            stroke = np.expand_dims(stroke, axis=0)
            
            if strokes is None:
                strokes = stroke
                
            else:
                strokes = np.concatenate([strokes, stroke], axis = 0)
             
        return strokes

##########################################################################################
# Data loader
##########################################################################################
class MezrabMotionDataset(Dataset):
    def __init__(self, dataset, device= None, flipped_matrix= False):
        """
        A custom-made dataloader to load data to the ML model
        """
        if device is None:
            device = 'cpu'
            
        self.dataset = dataset
        
        self.X_scaled = None
        self.X_centered = None
        self.X_centered_scaled = None
        
        self.X_np = self.dataset[:, :, :-1]
        self.Hand_np = self.dataset[:, :, -1:]
        
        self.X = torch.from_numpy(self.X_np).float().to(device)
        
        self.scaled_and_centered()
        
        self.Hand = torch.from_numpy(self.Hand_np).float().to(device)
        if flipped_matrix :
            self.data = {"X" : self.X.view(),
                        "Y" : self.Hand,
                        "X_scaled" : self.X_scaled,
                        "X_centered" : self.X_centered,
                        "X_centered_scaled" : self.X_centered_scaled,
                        "min_val" : self.min_val, #5
                        "max_val" : self.max_val, #6
                        "centered_min_val" : self.centered_min_val, #7
                        "centered_max_val" : self.centered_max_val #8
                        }
        else:
            self.data = {"X" : self.X,
                        "Y" : self.Hand,
                        "X_scaled" : self.X_scaled,
                        "X_centered" : self.X_centered,
                        "X_centered_scaled" : self.X_centered_scaled,
                        "min_val" : self.min_val, #5
                        "max_val" : self.max_val, #6
                        "centered_min_val" : self.centered_min_val, #7
                        "centered_max_val" : self.centered_max_val #8
                        }
        

    def  scaled_and_centered(self):
        """Scales the data between zero and one
        """
        self.min_val, self.max_val = get_min_max_from_dataset(self.X)
        
        self.X_scaled = torch.zeros_like(self.X)
        self.X_centered = torch.zeros_like(self.X)
        self.X_centered_scaled = torch.zeros_like(self.X)
        
        # finding the touching point of each motion and centering the motion on that
        center_points = torch.zeros_like(self.X[:, 0:1, :])        
        center_points[:, 0, :3] = self.X[:, 9, :3]
        self.X_centered =  self.X - center_points

        # scaling data between 0 and 1
        self.X_scaled =(self.X - self.min_val) / (self.max_val - self.min_val)

        # scaling the centered data between 0 and 1
        self.centered_min_val, self.centered_max_val = get_min_max_from_dataset(self.X_centered)
        self.X_centered_scaled = (self.X_centered - self.centered_min_val) / (self.centered_max_val - self.centered_min_val)

    
    def __len__(self):
        """
        Default behavior when len is called
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Default behavior when an item of the dataset is being called,
        it returns a dictionary, that contains all the data. Not super efficient
        in the case of large datasets, but pretty practical in the case of this
        model and small dataset that I have.
        """
        return {"X" : self.X[idx],
                "Y" : self.Hand[idx], 
                "X_scaled" : self.X_scaled[idx], 
                "X_centered" : self.X_centered[idx], 
                "X_centered_scaled" : self.X_centered_scaled[idx],
                "min_val" : self.min_val, #[0],
                "max_val" : self.max_val, #[0],
                "centered_min_val" : self.centered_min_val, #[0], 
                "centered_max_val" : self.centered_max_val, #[0]
                }  
        

##########################################################################################
# Visualization functinos
##########################################################################################    
def test_data_dimensions(strokes, list_of_features, w= 6):
    """Using a t-SNE algorithm this function process the data and plot them in a 2d manifold plot.
    It helps giving the user a better understanding of the data distribution.

    Args:
        strokes (nparray): an np array containing all the motion data in the shape of n x 20 x 10
        list_of_features (list): A list of indices of features we want to use in t-SNE algorithm, i.e., [0, 6, 7, 8]
        w (int, optional): setting the number of workers for the t-SNE algorithm. 
                           On my CPU, it seeems that 6 works the best.Defaults to 6.
    """
    # select features
    train_data= strokes[:,:,list_of_features]
    train_data = train_data.reshape(strokes.shape[0],-1)

    # fit the T-SNE algorithm
    tsne = TSNE(
                perplexity=30,
                metric="euclidean",
                n_jobs=w,
                random_state=42,
                verbose=False,
                )
    embedding_data = tsne.fit(train_data)
    
    # plot
    fig = generate_plots(embedding_data, strokes, list_of_features)
    fig.show()
    
def generate_plots(embedding_data, strokes, list_of_features):
    """Prepares the plot figure using the plotly go object.

    Args:
        embedding_data (numpy array): output of the t-SNE algorithm with shape of n x 2
        strokes (nparray): an np array containing all the motion data in the shape of n x 20 x 10
        list_of_features (list): A list of indices of features we want to use in t-SNE algorithm,
                                i.e., [0, 6, 7, 8]. It will be used to make the plot title

    Returns:
        plotly go.Figure: a figure with two traces, one for the left hand and one for the right hand
    """
    # setup
    feature_keys = ['px', 'py', 'pz', 'v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'hand']
    color_set = px.colors.qualitative.Plotly
    
    left_indices = np.where(strokes[:,0,-1]==0)
    right_indices = np.where(strokes[:,0,-1]==1)

    # making the plot
    fig = go.Figure()
    
    # making the two scatter plots
    left_hand_scatter = go.Scatter(
                            x= embedding_data[:, 0][left_indices],
                            y= embedding_data[:, 1][left_indices],
                            mode='markers',
                            name="Left",
                            text="Left",
                            marker=dict(size= 5,
                                        color= color_set[0],
                                        ),
                            )
                
    right_hand_scatter = go.Scatter(
                            x= embedding_data[:, 0][right_indices],
                            y= embedding_data[:, 1][right_indices],
                            mode='markers',
                            name="Right",
                            text="Right",
                            marker=dict(size= 5,
                                        color= color_set[-1],
                                        ),
                            )
                
    # combining the two in one plot 
    data = [left_hand_scatter, right_hand_scatter]
    fig = go.Figure(data = data)
    
    # adding the decorations and making the style!
    fig.update_layout(
                    coloraxis_showscale=False, 
                    margin=dict(l=5, r=5, t=50, b=5),
                    paper_bgcolor="rgba(39,39,39,255)",
                    plot_bgcolor="rgba(39,39,39,255)",
                    )
        
    # more here: https://plotly.com/python/axes/
    fig.update_xaxes(showticklabels=False, 
                        showgrid=False, 
                        zeroline=True,
                        zerolinewidth=1, 
                        zerolinecolor='gray',
                        fixedrange= True,)

    fig.update_yaxes(showticklabels=False, 
                        showgrid=False, 
                        zeroline=True,
                        zerolinewidth=1, 
                        zerolinecolor='gray', 
                        fixedrange= True,
                        )
    plot_title =  [feature_keys[i]+", " for i in list_of_features]
    plot_title = "".join(plot_title)
    plot_title = "Features: {}".format(plot_title)

    # more here: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    fig.update_layout(dragmode="lasso",
                        # selectdirection="d",
                        # clickmode= "event+select",
                        showlegend= True,
                        newshape_line_color="white",
                        newshape_line_width=1,
                        title=plot_title,
                        font=dict(
                                family="Roboto, monospace",
                                size=12,
                                color="white"
                                )
                    )
    return fig

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # folder to read data from
    csv_folder_path = "./data/motion_csv_files"

    # variables to adjust for data segmentation
    tresh_l = 0.289 # good for all
    tresh_h_normal = 0.4 # good for everything except Riz
    tresh_h_riz = 0.27 # only use this for Riz
    dist = 15 # good for all
    peak_dist = 30 # good for all, but if set on 35, it will be good for special cases
    motion_fixed_length = 20 # good for all

    # Process normal notes
    # reads the csv files and converts them to a pandas dataframe
    strokes_df = read_motion_csv_files(csv_folder_path, [0,1,2,3,4,7,8,9,10,11,12]) 
    # reads the dataframe and converts it ontp slices of Mezrab strokes
    strokes_processor = MotionSignalProcess(strokes_df, tresh_l, tresh_h_normal, dist, peak_dist, motion_fixed_length)
    # grabs all the final strokes from the object
    strokes = strokes_processor.strokes

    # process riz notes
    strokes_df_riz = read_motion_csv_files(csv_folder_path, [5,6]) 
    strokes_processor_riz = MotionSignalProcess(strokes_df_riz, tresh_l, tresh_h_riz, dist, peak_dist, motion_fixed_length)
    strokes_riz = strokes_processor_riz.strokes

    # stack the two set
    strokes = np.vstack([strokes, strokes_riz])

    # makes the dataoader out of the strokes
    mezrab_stroke_dataset = MezrabMotionDataset(strokes, device)
    train_dataset, valid_dataset, test_dataset, train_iterator, valid_iterator, test_iterator = dataloader_creator(mezrab_stroke_dataset)

    # test the visualizations
    test_data_dimensions(strokes, [6,7,8])
            