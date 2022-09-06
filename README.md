# Attentive-neural-networks-for-news-classification

* The goal of this this project is to build a news classification neural network model which can classify a hierarchial news dataset. Along with this, we propose a novel algorithm to detect class overlap in news categories, wihc can be learnt from computation on our model's intermediate stages.

## Requirments to run :

Inside the created Python environment (conda, venv, py-env etc.), run

    $ pip install -r requirements.txt

## GPU requirements :

* A GPU (CUDA enabled) is required to run the project. Average running time for main model on a single 12 GB GPU for us was ~7 hours.
* We provide a detailed method to setup PyTorch and Torchvision inside a Conda environment with CUDA support.

* Follow these steps:

    + Use the script [`conda_install.sh`](conda_install.sh)
    + Make it executable with `chmod +x ./install_conda.sh`
    + Run with `./install_conda.sh`

* The installation creates the following two directories in the install location:

    + `conda`: Contains the miniconda installation

    + `conda_pkgs`: Contains the cache for downloaded and decompressed packages

* Creating the first environment creates an additional directory in the install location:

    + `conda_envs`: Contains the created environment(s)

* In `conda/bin/` run `./conda init`

* Now run `conda create --name <custom_name> pytorch torchvision cudatoolkit=10.1 --channel pytorch` (this is our specific version)

* Run `conda activate <custom_name>` to activate the environment 
* Run `conda deactivate` to deactivate the current environment 
* Install missing requirements mentioned abobe with `pip install <package> >= <version>`

## Dataset

* [`preprocessing.py`](preprocessing.py) generates the required data files. 
* Base dataset with all news categories is stored in `news_data_test` in form of `.txt` files for every category.
* Updated dataset with less categories is stored in `news_data_test2` in the same way.

* These folders are not available in the repository anymore, to reduce storage size. They can be downloaded from [`here`](https://drive.google.com/drive/folders/1ursXywn-R9P9t-DU3SMy6s359kZsGOSN?usp=sharing) on request.

## Training and validating the model:

* To train the base model on all news classes , run [`main_base.py`](main_base.py). If there is no saved model, it will train and automatically save the model. 

* To train the base model on reduced news classes , run [`main_new.py`](main_new.py). If there is no saved model, it will train and automatically save the model. In the next run, it will automatically load the saved model instead of retraining.

* Training and validation metrics are saved in [`values_base.txt`](values_base.txt) for the base model.
* Training and and validation metrics are saved in [`values_new.txt`](values_new.txt) for the new model.

* Plots for the base model are in [`data_figs_base`](data_figs_bas).
* Plots for the updated model are in [`data_figs_new`](data_figs_new).

## Visualizing the similarity algorithm:

* Run [`overlap_algo.py`](overlap_algo.py) to visualize the class overlap on the base dataset. tSNE values are generated from `main_base.py` and will be stored in `tSNE_values.csv`. The results are available [`here`](https://drive.google.com/drive/folders/1ursXywn-R9P9t-DU3SMy6s359kZsGOSN?usp=sharing)
* Thresholds can be changed in the file to adapt the algorithm.


## Other files:

* [`model.py`](model.py) contains our transformer model defintion.
* [`helper.py`](helper.py) contains utility methods for dataset loading for the model, evaluation metric methods etc.

## Presentation slides: 
* [`slides_final`](slides_final.pdf)

## Written thesis:
* [`thesis.pdf`](thesis.pdf)

## Note :

* To change the dataset from base to updated, you need to change the dataset source in `helper.py` from `news_data_test` to `news_data_test2` and vice-versa.
* We have the preliminary implementation of our RNN model in `RNNMODEL`. This requires some pre-requisites to be installed which are not `Pypi` standard. To install those, we 
recommend you visit the original source [InferSent](https://github.com/facebookresearch/InferSent)















