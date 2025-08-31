import random

import numpy as np
import matplotlib.pyplot as plt

def create_histogram(data: np.array,plot_file: str=None,hist_type: str="bar"):
    plt.title("histogram of random values",loc="center")
    bins = np.arange(data.min(),data.max(),step=0.5)
    plt.hist(data,color="green",bins=bins,histtype=hist_type)
    if plot_file is not None:
        plt.savefig(plot_file)

def create_multiple_hist(data: np.array, plot_file: str=None):
    fig = plt.figure()
    fig,ax = plt.subplots(1,data.shape[1])
    fig.suptitle("Horizontally stacked histograms")
    for i in range(data.shape[1]):
        color = random.choice(["red","green","blue","yellow"])
        bins = np.arange(data.min(),data.max(),step=0.5)
        ax[i].hist(data[:,i],bins=bins,color=color)
        ax[i].set_title(f"histogram {i+1}")
    if plot_file is not None:
        plt.savefig(plot_file)


# hist_data = np.random.randn(500)
# create_histogram(hist_data,plot_file="img/hist_stepfilled.png",hist_type="stepfilled")

multi_hist_data = np.random.randn(500,4)
create_multiple_hist(multi_hist_data, plot_file="img/stacked_hist_hor.png")