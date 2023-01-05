# ThirdHand
![demo](media\visualizations\thirdHand_hero_image.jpg)
---

ThirdHand is a case study as part of my  PhD research on *A Situated Collaborative Framework for
Machine Learning-Based Toolmaking for Creative Practitioners*.  

## Summary
In this casestudy, I collaborated with a mucsician to create a robtic musical instrument based on the user-generated data. A generative model was trained on the collected data from the musician to control a robotic arm to play Santoor, a traditional Persian musical instrument. 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mLxlOaPRUNs/0.jpg)](https://www.youtube.com/watch?v=mLxlOaPRUNs)


![demo](media\visualizations\Picture1.png)


## Data Pipeline
![demo](media\visualizations\63_a.png)
![demo](media\visualizations\59.jpg)

## Data Post Processing
![demo](media\visualizations\72.png)
![demo](media\visualizations\73.png)

## ML Models
![demo](media\visualizations\75.png)
![demo](media\visualizations\74.png)

## Model's Performance
### Reconstruction
![demo](media\visualizations\78.png)
### Generation
![demo](media\visualizations\79_c.png)

## Robotic Setup
![demo](media\visualizations\80.png)
![demo](media\visualizations\83.png)
![demo](media\visualizations\81_a.jpg)
![demo](media\visualizations\81_b.jpg)

## Demo
![demo](media\visualizations\motion_type.gif)
![demo](media\visualizations\87.jpg)


## Requirements:
### Base:
* Python: 3.10.8
* CUDA: 11.6

### Dependencies:
Use the [spec-file.txt](spec-file.txt) to reproduce the Conda environment. The main libraries are listed below:
* [PyTorch](https://pytorch.org/get-started/locally/) 1.13.1 <br>
    ``` conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia```
* [Tensorboard](https://www.tensorflow.org/tensorboard) 2.11.0 <br>
    ```conda install -c conda-forge tensorbaord```
* [WandB](https://wandb.ai/site) 0.13.7 <br>
    ```conda install wandb --channel conda-forge```
* [Plotly](https://plotly.com/) 5.11.0 <br>
    ```conda install -c plotly plotly```
* [Scikit-Learn](https://scikit-learn.org/) 1.1.2 <br>
    ```conda install -c conda-forge scikit-learn```
* [Pandas](https://pandas.pydata.org/) 1.5.2 <br>
    ```conda install pandas```
* [OpenTSNE](https://opentsne.readthedocs.io/en/latest/index.html) 0.6.2 <br>
    ```conda install --channel conda-forge opentsne```
* [PeakUtil](https://peakutils.readthedocs.io/en/latest/#installation) 1.3.3
    * Download the files from this [link](https://zenodo.org/record/887917#.Y7RLnnbMIuU),
    * Unzip the package and navigate to the folder,
    * Run the srtup file:<br>
    ```python setup.py isntall```
* [Kaleido](https://pypi.org/project/kaleido/#description) <br>
    ```conda install -c conda-forge python-kaleido```
* [NBFormat](https://pypi.org/project/nbformat/) <br>
    ```conda install -c conda-forge nbformat```