import os
import time
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import networks as networks
from collections import OrderedDict
import matplotlib.pyplot as plt
import dload
from utils import *
import shared_networks as shared_networks
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

sns.set_style("dark")
sns.set_palette("colorblind")
sns.set_theme()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 8, figsize=(10,5), gridspec_kw={'width_ratios': [1,3,1]})
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(1, 9, figsize=(20,5), gridspec_kw={'width_ratios': [3,1,1,1,1,1,1,1,1]})
# color_palette = [(86, 100, 26), (192, 175, 251), (230, 161, 118), (94, 190, 171), (152, 68, 100), (0, 103, 138), (205, 205, 205)]




width = 0.2
margin = 0.015
font_size = 16
color_palette = [(94, 190, 171), (230, 161, 118), (0, 103, 138), (205, 205, 205)]
color_palette = np.array(color_palette) / 255


## JS is multiplied with 100
## MI is divided by 100
labels = ['DBI Train', 'DBI Val', 'DBI Test', 'JS', 'Hellinger', 'Bhattacharyya', 'Pearson', 'Mutual Info', 'ECE', 'MCE', 'Face Entropy']
CE = [1.38, 1.77, 2.04, 0.00071  , 0.1554 , 0.0624 , 0.1002 , 5.508, 7.030000 , 48.620000 , 0.5880]
KD = [1.55, 1.87, 2.04 , 0.00050  , 0.1247 , 0.0497 , 0.0360 , 5.482, 7.470000 , 52.600000 , 0.5880]
CKD = [ 1.46, 1.69, 1.82 , 0.000079  , 0.0590 , 0.0092 , 0.00081 , 5.562, 7.160000 , 47.260000 , 0.6497]


## DBI
x = np.arange(len(labels[:3]))  # the label locations
rects1 = ax1.bar(x - width - margin, CE[:3], width, label='CE', color=color_palette[0])
rects2 = ax1.bar(x , KD[:3], width, label='KD', color=color_palette[1])
rects3 = ax1.bar(x + width + margin, CKD[:3], width, label='CKD', color=color_palette[2])
# ax2.set_ylabel('Scores')
ax1.tick_params(axis='both', which='major', labelsize=font_size)
ax1.set_xticks(x, labels[:3], fontsize=font_size)
ax1.set_ylim((1.2, 2.2))
ax1.legend(loc='upper left', fontsize=14)

# ax1.bar_label(rects1, padding=3, fontsize=9)
# ax1.bar_label(rects2, padding=3, fontsize=9)
# ax1.bar_label(rects3, padding=3, fontsize=9)


## JS
x = np.arange(len(labels[3:4]))  # the label locations
rects0 = ax2.bar(x - width - margin - width/2, 0, width)
rects5 = ax2.bar(x + width + margin + width/2, 0, width)
rects1 = ax2.bar(x - width - margin, CE[3:4], width, label='CE', color=color_palette[0])
rects2 = ax2.bar(x , KD[3:4], width, label='KD', color=color_palette[1])
rects3 = ax2.bar(x + width + margin, CKD[3:4], width, label='CKD', color=color_palette[2])
# ax2.set_ylabel('Scores')
ax2.tick_params(axis='both', which='major', labelsize=font_size)
ax2.set_xticks(x, labels[3:4])
ax2.set_ylim((0, 0.00075))

# ax2.bar_label(rects1, padding=3, fontsize=8)
# ax2.bar_label(rects2, padding=3, fontsize=8)
# ax2.bar_label(rects3, padding=3, fontsize=8)

## Hellinger
x = np.arange(len(labels[4:5]))  # the label locations
rects0 = ax3.bar(x - width - margin - width/2, 0, width)
rects5 = ax3.bar(x + width + margin + width/2, 0, width)
rects1 = ax3.bar(x - width - margin, CE[4:5], width, label='CE', color=color_palette[0])
rects2 = ax3.bar(x , KD[4:5], width, label='KD', color=color_palette[1])
rects3 = ax3.bar(x + width + margin, CKD[4:5], width, label='CKD', color=color_palette[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')
ax3.tick_params(axis='both', which='major', labelsize=font_size)
ax3.set_xticks(x, labels[4:5])
ax3.set_ylim((0.04, 0.17))
# ax2.legend(loc='upper right')

# ax3.bar_label(rects1, padding=3, fontsize=9)
# ax3.bar_label(rects2, padding=3, fontsize=9)
# ax3.bar_label(rects3, padding=3, fontsize=9)

## Bhatt
x = np.arange(len(labels[5:6]))  # the label locations
rects0 = ax4.bar(x - width - margin - width/2, 0, width)
rects5 = ax4.bar(x + width + margin + width/2, 0, width)
rects1 = ax4.bar(x - width - margin, CE[5:6], width, label='CE', color=color_palette[0])
rects2 = ax4.bar(x , KD[5:6], width, label='KD', color=color_palette[1])
rects3 = ax4.bar(x + width + margin, CKD[5:6], width, label='CKD', color=color_palette[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')
ax4.tick_params(axis='both', which='major', labelsize=font_size)
ax4.set_xticks(x, labels[5:6])
ax4.set_ylim((0.005, 0.07))
# ax2.legend(loc='upper right')

# ax4.bar_label(rects1, padding=3, fontsize=9)
# ax4.bar_label(rects2, padding=3, fontsize=9)
# ax4.bar_label(rects3, padding=3, fontsize=9)

## Pearson
x = np.arange(len(labels[6:7]))  # the label locations
rects0 = ax5.bar(x - width - margin - width/2, 0, width, bottom=0.001)
rects5 = ax5.bar(x + width + margin + width/2, 0, width, bottom=0.001)
rects1 = ax5.bar(x - width - margin, CE[6:7], width, label='CE', color=color_palette[0], bottom=0.001)
rects2 = ax5.bar(x , KD[6:7], width, label='KD', color=color_palette[1], bottom=0.001)
rects3 = ax5.bar(x + width + margin, CKD[6:7], width, label='CKD', color=color_palette[2], bottom=0.001)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')

ax5.tick_params(axis='both', which='major', labelsize=font_size)
ax5.set_yscale('log')
ax5.set_xticks(x, labels[6:7])

# ax5.bar_label(rects1, padding=3, fontsize=9)
# ax5.bar_label(rects2, padding=3, fontsize=9)
# ax5.bar_label(rects3, padding=3, fontsize=8)

## MI
x = np.arange(len(labels[7:8]))  # the label locations
width = 0.1  # the width of the bars
rects0 = ax6.bar(x - width - margin - width/2, 0, width)
rects5 = ax6.bar(x + width + margin + width/2, 0, width)
rects1 = ax6.bar(x - width - margin, CE[7:8], width, label='CE', color=color_palette[0])
rects2 = ax6.bar(x , KD[7:8], width, label='KD', color=color_palette[1])
rects3 = ax6.bar(x + width + margin, CKD[7:8], width, label='CKD', color=color_palette[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')
ax6.tick_params(axis='both', which='major', labelsize=font_size)
ax6.set_xticks(x, labels[7:8])
ax6.set_ylim((5.45, 5.58))
# ax3.legend()

# ax6.bar_label(rects1, padding=3, fontsize=10)
# ax6.bar_label(rects2, padding=3, fontsize=10)
# ax6.bar_label(rects3, padding=3, fontsize=10)




### ECE, MCE


width = 0.1  # the width of the bars
margin = 0.015
## ECE
x = np.arange(len(labels[8:9]))  # the label locations
width = 0.1  # the width of the bars
rects0 = ax7.bar(x - width-margin - width/2, 0, width)
rects5 = ax7.bar(x + width+margin + width/2, 0, width)
rects1 = ax7.bar(x - width-margin, CE[8:9], width, label='CE', color=color_palette[0])
rects2 = ax7.bar(x , KD[8:9], width, label='KD', color=color_palette[1])
rects3 = ax7.bar(x + width+margin, CKD[8:9], width, label='CKD', color=color_palette[2])


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')
ax7.tick_params(axis='both', which='major', labelsize=font_size)
ax7.set_xticks(x, labels[8:9])
ax7.set_ylim((5.5, 7.7))
# ax1.legend()

# ax7.bar_label(rects1, padding=3, fontsize=10)
# ax7.bar_label(rects2, padding=3, fontsize=10)
# ax7.bar_label(rects3, padding=3, fontsize=10)
# ax7.bar_label(rects4, padding=3, fontsize=10)

## MCE
x = np.arange(len(labels[9:10]))  # the label locations
width = 0.1  # the width of the bars
rects0 = ax8.bar(x - 1.5*width-margin - width/2, 0, width)
rects5 = ax8.bar(x + 1.5*width+margin + width/2, 0, width)
rects1 = ax8.bar(x - 1.5*width-margin, CE[9:10], width, label='CE', color=color_palette[0])
rects2 = ax8.bar(x - 0.5*width-0.005, KD[9:10], width, label='KD', color=color_palette[1])
rects3 = ax8.bar(x + 0.5*width+0.005, CKD[9:10], width, label='CKD', color=color_palette[2])



# Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Scores')
ax8.tick_params(axis='both', which='major', labelsize=font_size)
ax8.set_xticks(x, labels[9:10])
ax8.set_ylim((40, 54))
# ax1.legend()


# ax8.bar_label(rects1, padding=3, fontsize=9)
# ax8.bar_label(rects2, padding=3, fontsize=9)
# ax8.bar_label(rects3, padding=3, fontsize=9)
# ax8.bar_label(rects4, padding=3, fontsize=9)



CE = [0.588]
KD = [0.588]
CKD = [0.6497]



x = np.arange(len([0]))  # the label locations
width = 0.1  # the width of the bars

rects0 = ax9.bar(x - width-margin - width/2, 0, width)
rects5 = ax9.bar(x + width+margin + width/2, 0, width)
rects1 = ax9.bar(x - width-margin, CE[0], width, label='CE', color=color_palette[0])
rects2 = ax9.bar(x, KD[0], width, label='KD', color=color_palette[1])
rects3 = ax9.bar(x + width+margin, CKD[0], width, label='CKD', color=color_palette[2])
ax9.tick_params(axis='both', which='major', labelsize=font_size)
ax9.set_xticks(x, ['Face Entropy'])
ax9.set_ylim([0.5, 0.7])
# ax9.legend(loc='upper left', fontsize=16)








fig.tight_layout()





plt.savefig('./barplot_all_final.pdf')



