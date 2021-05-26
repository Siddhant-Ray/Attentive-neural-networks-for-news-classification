# Attentive-neural-networks-for-news-classification

* The goal of this this project is to build a news classification neural network model which can classify a hierarchial news dataset. Along with this, we propose a novel algorithm to detect class overlap in news categories, wihc can be learnt from computation on our model's intermediate stages.

## Requirments to run :

* Python >= 3.8.0
* PyTorch >= 1.7.0
* Torchvision >= 0.8.0
* Jupyter (optional for testing only - develop branch ) >= 1.0.0
* Transformers >= 4.3.0
* Matplotlib >= 3.3.0
* Scipy >= 1.6.0
* Scikit-learn >= 0.24.0
* Pandas >= 1.2.0
* Seaborn >= 0.11.0

## GPU requirements :

* A GPU (CUDA enabled) is required to run the project. Average running time for main model on a single 12 GB GPU for us was ~7 hours.
* We provide a detailed method to setup PyTorch and Torchvision inside a Conda environment with CUDA support.

* Follow these steps in order :

1. Use the script conda_install.sh
2. Make it executable with chmod +x ./install_conda.sh
3. Run with ./install_conda.sh

* The installation creates the following two directories in the install location:

    + conda: Contains the miniconda installation

    + conda_pkgs: Contains the cache for downloaded and decompressed packages

* Creating the first environment creates an additional directory in the install location:

    + conda_envs: Contains the created environment(s)

* In conda/bin/, run .conda init

* Now run conda create --name <custom_name> pytorch torchvision cudatoolkit=10.1 --channel pytorch (this is our specific version)

* conda activate <custom_name> to activate the environment 
* conda deactivate to deactivate the current environment 
* install missing requirements mentioned aobe with pip install <package>>=<version>











