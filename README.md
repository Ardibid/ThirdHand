# ThirdHand

---
## Requirements:
### Base:
* Python: 3.10.8
* CUDA: 11.6

### Dependencies:
* [PyTorch](https://pytorch.org/get-started/locally/): 1.13.1 <br>
    ``` conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia```
* [WandB](https://wandb.ai/site): 0.13.7 <br>
    ```conda install wandb --channel conda-forge```
* [Plotly](https://plotly.com/): 5.11.0 <br>
    ```conda install -c plotly plotly```
* [Scikit-Learn](https://scikit-learn.org/): 1.1.2 <br>
    ```conda install -c conda-forge scikit-learn```
* [Pandas](https://pandas.pydata.org/): 1.5.2 <br>
    ```conda install pandas```
* [OpenTSNE](https://opentsne.readthedocs.io/en/latest/index.html): 0.6.2 <br>
    ```conda install --channel conda-forge opentsne```
* [PeakUtil](https://peakutils.readthedocs.io/en/latest/#installation): 1.3.3
    * Download the files from this [link](https://zenodo.org/record/887917#.Y7RLnnbMIuU),
    * Unzip the package and navigate to the folder,
    * Run the srtup file:<br>
    ```python setup.py isntall```