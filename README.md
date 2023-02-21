# ThirdHand
<p align="center">
    <img src="./media/thirdHand_hero_image.jpg" style="border-radius:20px;"/>
</p>
Ardavan Bidgoli &copy;

ThirdHand is a case study as part of my  PhD research on "*A Collaborative Framework for Machine Learning-Based Toolmaking for Creative Practitices*". This study proposes a framework for making robotic musical instruments to augment an artist’s capability to play santur. It is not destined to replace the artist, but it is an effort to explore the affordances of machine learning for musical toolmaking. <br>
This project utilizes a dataset of six-degree-of-freedom motions provided by the musician as a vehicle to convey the specific idiom of the artist. The samples presented earlier in this chapter did not focus on this aspect.<br>


## Summary
This study focuses on developing a robotic musical instrument to play santur based on the samples provided by the participating musician. Accordingly, composing, generating, or improvising music is out of the scope of this study. The data modality is restricted to mezrab strokes, encoded as sequences of six-degree-of-freedom motions. <br>

<br>

A Conditional Variational AutoEncoder was trained on the mezrab motions demonstrated by the musician. We want to see if it is possible to generate  novel mezrab stroke samples that closely resembles hers, but with slight variation, and then play them on a robotics arm. The model does not generate or compose new musical scores and the robotic musical instrument does not behaves interactively.  

<p align="center">
    <img src="./media/general_workflow.png" width="80%"/>
</p>


## Using the Model

To train the model, use [thirdhand_trainer.py](./thirdhand_trainer.py). The default hyper-parameters are extracted from the  on the WandB optimization results: <br>

```python thirdhand_trainer.py -mode=train -epochs=150```

You can alternatively load the train model and see its generated motions: <br>
```python thirdhand_trainer.py -mdoe=load```
<br>
To optimize the model using WandB, use the [optimizer boiler plate notebook](./Optimization_boilerplate.ipynb).

## Data Pipeline
*Santur* is a traditional Persian stringed musical instrument with common roots with the hammered dulcimer. The instrument is usually played while putting stationary on an inclined platform. Its trapezoidal frame seats in front of the musician and provides a flat framework for its 72 strings. The player uses two wooden hammers (*Mezrab*) to play the instrument.

The data samples are 6-DoF motiones of mezrabs (hammers). Each mezrab was tagged with reflective tapes to allow the motion capture system register their motion @120Hz.

<p align="center">
    <img src="./media/trackers.png" width="80%" style="border-radius:10px;"/>
    <img src="./media/playing_santoor.gif" width="80%" style="border-radius:10px;"/>
</p>

The stream of motion capture data was later cleaned saved as csv fiels. The functions available in *thirdHand_data_loader.py* automatically sliced the data into shorter sequences and assing the left/right hand labels. The samples are scaled into [0, 1] and centered around the (0,0,0). 
Each motion each sample was formatted as a $20 × 9$ vector, representing the 20 poses in space, each defined by a point and two vectors.
$$X_t = [x_0, x_1, ..., x_{19}]$$
$$x_i = [px_i, py_i, pz_i, v_xx_i, v_xy_i, v_xz_i, v_yx_i, v_yy_i, v_yz_i]$$

<p align="center">
    <img src="./media/signal_processing.png"width="60%" style="border-radius:10px;"/>
    <img src="./media/motion_breakdown.png" width="60%" style="border-radius:10px;"/>
</p>


<p align="center">
    <img src="./media/motion_raw_quick.gif" width="60%" style="border-radius:10px;"/>
</p>
<p align="center">
    <img src="./media/motions_sampler_of_250.png" width="60%"/>
</p>

## ML Models

The model is based on the general Conditional Variational AutoEncoder architecture. he conditioning signal is a one-hot vector of dim 1x2 that determines the hand label.

<p align="center">
   <img src="./media/C_VAE_Architecture.png" width="80%"/>
</p>

The temporal features of the mezrab's motions werer captured using 1-D CNN layers. T
<p align="center">
   <img src="./media/1D_CNN.png" width="80%"/>
</p>

The model was optimized using WandB optimizer.
<p align="center">
   <img src="./media/wandb.png" width="80%"/>
</p>

### Model's Performance

The model can reconstruct the motions after around 150 epochs and it can be used to generate new motions by sampling from the latent space.

<p align="center">
   <img src="./media/training_process.png" width="80%"/>
</p>

<p align="center">
   <img src="./media/generated_samples.gif" width="60%" style="border-radius:10px;"/>
   
</p>

## Robotic Setup

The generated motions were post-processed using *Grasshopper* and *HAL* plugin to control a robotic arm, equipped with a mezrab. 
<p align="center">
   <img src="./media/robot_setup.png" width="80%"/>
</p>

<div align="center">
    <p float="left">
            <img src="./media/holder.jpg" width="32%" />
            <img src="./media/mezrab_on_robot.jpg" width="32%" /> 
            <img src="./media/robot_on_santoor.jpg" width="14.5%" /> 
    </p>
</div>

<p align="center">
   <img src="./media/motion_type.gif" width="80%"/>
</p>

## Demo

On the left side video, the robot plays notes based on the motion generated by the C-VAE model, on the right side, the robot just repeats a single-joint motion to strike the strings.

<div align="center">
    <p float="left">
            <img src="./media/playing_duet.gif" width="80%" style="border-radius:10px;"/>
    </p>
</div>
<p align="center">
   <img src="./media/motion_type.gif" width="80%"/>
</p>

[![ThirdHand Video](https://img.youtube.com/vi/Vyp7q1vxXnw/0.jpg)](https://www.youtube.com/watch?v=Vyp7q1vxXnw)


## Requirements and Dependencies:
* Python: 3.10.8
* CUDA: 11.6

Use the [spec-file.txt](spec-file.txt) to reproduce the Conda environment. The main libraries are listed below:
* [PyTorch](https://pytorch.org/get-started/locally/) 1.13.1 
    ``` conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia```
* [Tensorboard](https://www.tensorflow.org/tensorboard) 2.11.0
    ```conda install -c conda-forge tensorbaord```
* [WandB](https://wandb.ai/site) 0.13.7 
    ```conda install wandb --channel conda-forge```
* [Plotly](https://plotly.com/) 5.11.0
    ```conda install -c plotly plotly```
* [Scikit-Learn](https://scikit-learn.org/) 1.1.2
    ```conda install -c conda-forge scikit-learn```
* [Pandas](https://pandas.pydata.org/) 1.5.2
    ```conda install pandas```
* [OpenTSNE](https://opentsne.readthedocs.io/en/latest/index.html) 0.6.2
    ```conda install --channel conda-forge opentsne```
* [PeakUtil](https://peakutils.readthedocs.io/en/latest/#installation) 1.3.3
    * Download the files from this [link](https://zenodo.org/record/887917#.Y7RLnnbMIuU),
    * Unzip the package and navigate to the folder,
    * Run the srtup file:
    ```python setup.py isntall```
* [Kaleido](https://pypi.org/project/kaleido/#description)
    ```conda install -c conda-forge python-kaleido```
* [NBFormat](https://pypi.org/project/nbformat/)
    ```conda install -c conda-forge nbformat```

## Notes:
Developed as a part of Ardavan Bidgoli's PhD thesis @ Carnegie Mellon University, school of Architecture. In collaboration with Mahtab Nadalian. This project was partially supported by Computational Design research support microgrant @ CMU School of Architecture. <br>
If you are interested in robotic musical instrument design for Persian music, you will enjoy [Santoor Bot by Mohammad Jafari](https://music.gatech.edu/santoor-bot), as much as I did! The primary difference between the two projects is the focus on the musician's specific mezrab motions and using a deep learning model to generate new motions simialr to the one demonstrated by the musician. <br>

---
By Ardavan Bidgoli, 2021-2023, [GitHub](https://github.com/Ardibid)
