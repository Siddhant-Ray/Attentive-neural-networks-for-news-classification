import numpy as np
from pyemd import emd
from scipy.spatial.distance import jensenshannon
import pandas as pd
import seaborn as sns
from pylab import savefig

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

from helper import DatasetLoader

from pathlib import Path
import os

# Load the dataset using the helper class
data_set = DatasetLoader()

# Calculate the number of categories and get the dictionary mapping categories to news entries
number_of_categories, category_news  = data_set.get_categories()

### Dictionaries mapping caegory grom text to number and vice-versa
label_list = list(category_news.keys())
index_class_map_dict={}

index_class_map_dict1={}

for idx, value in enumerate(label_list):
    index_class_map_dict1[value]=idx

for idx, value in enumerate(label_list):
    index_class_map_dict[idx]=value

""" The following steps contain our algorithm
 which decides which classes are similar"""

def class_similarity_algorithm(): 

    ### Number of news categories, over here 38.
    tot_classes = number_of_categories

    df = pd.read_csv('tSNE_values.csv')
    print(df.head())

    ### Print t-SNE plots for all classes and fit 2D histograms for all
    save_dir = Path("tSNE_plots")
    if not save_dir.exists():
        save_dir.mkdir()

    hists = []
    hist_range = [[df['X'].min(), df['X'].max()], [df['Y'].min(), df['Y'].max()]]
    bins = 45
    for i in range(tot_classes):
        masked = df[df['Label'] == i]
        fig = plt.figure(figsize=(15,15))
        name = index_class_map_dict[i]
        plt.title(name, size = 40)
        plt.xticks(size = 40)
        plt.yticks(size = 40)
        plt.scatter(masked['X'], masked['Y'], 7)
        path_save = save_dir / f"{name}.png"
        fig.savefig(path_save)
        plt.close()
        
        curr_hist = np.histogram2d(masked['X'], masked['Y'], range=hist_range, bins=bins, normed=True)[0]
        hists.append(curr_hist.reshape(-1))
    hists = np.array(hists)


    ### Calculate pairwise JD divergence
    js_div = np.empty([tot_classes, tot_classes], dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for i in range(tot_classes):
        for j in range(tot_classes):
            js_div[i, j] = jensenshannon(hists[i] + eps, hists[j] + eps).sum()

    ### Plot JS divergence heatmap and histogram of its values
    df_js = pd.DataFrame(js_div, index = label_list,columns = label_list)
    plt.rcParams['figure.figsize'] = (20,20)
    matrix = sns.heatmap(df_js, annot=True, cmap='Blues')
    matrix.set_yticklabels(matrix.get_ymajorticklabels(), fontsize = 18)
    matrix.set_xticklabels(matrix.get_xmajorticklabels(), fontsize = 18)
    cbar = matrix.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)

    path = os.getcwd()
    folder = "data_figs_base"

    FILE = "jsdiv_full.png"

    path_save = os.path.join(path, folder)
    path_save = os.path.join(path_save, FILE)

    figure = matrix.get_figure()    
    figure.savefig(path_save)
    plt.show()


    FILE = "jenshan_hist_full.png"

    path_save = os.path.join(path, folder)
    path_save = os.path.join(path_save, FILE)

    fig = plt.figure(figsize=(15,15))
    js_div_new = [js_div[i, j] for i in range(tot_classes) for j in range(tot_classes) if i!=j]
    plt.hist(js_div_new, bins=35, density=True)
    plt.title("Histogram of JS Divergence values",size = 30)

    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.savefig(path_save)


    ### Decision for merging and removing of classes
    BASIC_MERGE_THRESH = 0.69
    JS_THRESH = 0.70
    COUNT_THRESH = 3

    mask = (df_js <= BASIC_MERGE_THRESH) & (df_js > 0)
    i, j = np.where(mask)
    idxs = np.array([(ii, jj) for ii, jj in zip(i, j) if ii < jj])
    to_merge = list(zip(df_js.index[idxs[:,0]], df_js.columns[idxs[:,1]]))

    print("Possibly merge these classes :")
    for c1, c2 in to_merge:
        print(f"  {c1} -- {c2}")

    count_js = (df_js < JS_THRESH).sum() - 1
    mask1 = count_js >= COUNT_THRESH
    i = np.where(mask1)
    to_drop = count_js.index[i]
    print("Multiple overlap: Possibly drop these classes:")
    for c in to_drop:
        print(f"  {c}")



    ### Define a custom colormap for all class tSNE combined plot
    gen = np.random.default_rng(6)
    rgb = gen.uniform(size=(tot_classes, 3))
    gen.shuffle(rgb)
    a = np.ones((tot_classes,1))
    rgba = np.concatenate([rgb, a], 1)
    custom_cmap = ListedColormap(rgba)


    ### Plot combined tSNE vectors for all classes
    path = os.getcwd()

    folder = "tSNE_plots"
    FILE = "clustered_tSNE.png"

    path_save = os.path.join(path, folder)
    path_save = os.path.join(path_save, FILE)


    fig = plt.figure(figsize=(15,15))
    out = plt.scatter(df['X'],df['Y'], 10, c = df['Label'], cmap=custom_cmap)
    cbar = plt.colorbar(out, ticks = np.arange(tot_classes))
    cbar.set_ticklabels(list(index_class_map_dict1.keys()))
    cbar.ax.tick_params(labelsize=14)
    plt.title("tSNE on pre-classified output vs ground truth labels", size = 30)
    plt.xticks(size = 25)
    plt.yticks(size = 25)

    file_name = path_save

    plt.savefig(file_name)
    plt.show()


if __name__ == '__main__':
    class_similarity_algorithm()




