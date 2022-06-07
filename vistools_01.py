import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import sem, t
from pathlib import Path
import nmrglue as ng
import matplotlib.pyplot as plt
%matplotlib inline

def get_data(fname):
    container = np.load(rootpath/'{}.npz'.format(fname))
    ms_data = [container[key] for key in container]
    ms_df = pd.DataFrame(ms_data).fillna(0)
    ms_df = ms_df.to_numpy()
    ms_label = pd.read_csv(rootpath/'{}.csv'.format(fname))
    
    return ms_label, ms_df


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)
      

def get_UBandLB(mean, std):
    ub = mean + std
    lb = mean - std
    return ub, lb


def ci095(data):
    confidence = 0.95
    n = len(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return h


def datainconditions(label, df):
    R_data = []
    S_data = []

    for i, rs in enumerate(label[label.columns[4]]):
        if rs == 1:
            R_data.append(df[i])
        else:
            S_data.append(df[i])
    return R_data, S_data


def plotPseudoGel(ConName, data, s1, s2):
#     fig, axes = plt.subplots(len(ConName), 1, figsize=(28, 4*len(ConName)), dpi=150)
    fig, axes = plt.subplots(len(ConName), 1, gridspec_kw={'height_ratios': [len(data[idx]) for idx in range(len(ConName))]}, figsize=(28, 4*len(ConName)), dpi=150)
#     fig.suptitle('MSSA/MRSA', fontsize=24)
    for idx in range(len(ConName)):
        axes[idx].set_title(ConName[idx], fontsize=20)
        im = axes[idx].imshow(np.array(data[idx])[:, (s1-2000):(s2-2000)], cmap='jet', vmin=0, vmax=1, interpolation='nearest', aspect='auto', extent=[s1, s2, 0, len(data[idx])])
    # colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(rootpath/'results_imgs/cKPamber_pseudogel{}_{}.png'.format(s1, dt.now().strftime('%Y%m%d')), dpi=150)
    plt.show()
    
    
def plotMSfig(ConName, data, s1, s2, ds):
#     color_list = [['k', 'gray'], ['b', 'lightblue'], ['r', 'pink'], ['g', 'lightgreen']]
    fig = plt.figure(figsize=(28, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    x_ticks = np.arange(s1, s2, ds)
    plt.xticks(x_ticks)
    ax.set_xlim(s1, s2)
    plt.ylim([0.0, 1.0])
    
    colors = plt.cm.jet(np.linspace(0, 1, len(ConName)))
    for idx in range(len(ConName)):
        h1 = ci095(data[idx])
        mean1 = np.average(data[idx], 0)
#         plt.plot(range(s1, s2), mean1[(s1-2000):(s2-2000)], color=colors[idx], label=ConName[idx])
        plt.errorbar(range(s1, s2), mean1[(s1-2000):(s2-2000)], yerr=h1[(s1-2000):(s2-2000)], 
                     color=colors[idx], ecolor=colors[idx], label=ConName[idx])

    plt.legend(loc='upper right', prop={'size': 20})
    plt.savefig(rootpath/'results_imgs/cKPamber_splot{}_{}.png'.format(s1, dt.now().strftime('%Y%m%d')), dpi=150)
