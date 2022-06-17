import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import networks as networks
import shared_networks as shared_networks
from collections import OrderedDict
import matplotlib.pyplot as plt
import dload
from utils import *
import json
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
from scipy import interp
from PIL import Image
import math
import itertools
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

exp = '01'

# dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
dset_list = ['ethnic']

for dset in dset_list:
    mat = np.load('./iden_top1_confusion_mats/exp' + exp + '/'+dset+'_mat.npy')
    print(mat)
    fmt = '.2f' 
    thresh = mat.max() / 2.
    top = mpl.cm.get_cmap('Oranges_r', 128)
    bottom = mpl.cm.get_cmap('Blues', 128)
    new_colors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    new_cmap = ListedColormap(new_colors, name="OrangeBlue")
    plt.imshow(mat, cmap=new_cmap)
    plt.colorbar()
    # for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
    #     plt.text(j, i, format(mat[i, j], fmt), horizontalalignment="center", color="white" if mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./iden_top1_confusion_mats/exp'+exp + '/' + dset+'_confusion_mat.jpg')
    plt.close()

    # for i in range(len(mat)):
    #     for j in range(len(mat)):
    #         if mat[i,j] == np.nan:
    #             print(11)


