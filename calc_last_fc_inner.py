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



# exp_list = ['01', '02', '04']
exp_list = ['01', '04']
epoch = '70'


mat_list = []

for exp in exp_list:
    print(exp)

    root_dir = os.path.join('results', 'exp_'+exp)
    model_dir = os.path.join(root_dir, 'models')
    figure_dir = os.path.join(root_dir, 'figures', epoch, 'ocular')


    model = shared_networks.shared_network(num_classes=1054)
    ## load model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_'+epoch+'.pth.tar'), map_location='cpu')['state_dict'])
    face_mat = model.face_fc.weight
    ocular_mat = model.ocular_fc.weight

    out = torch.abs(torch.matmul(face_mat, face_mat.t()) - torch.matmul(ocular_mat, ocular_mat.t())) / 2.0

    print('test')
    print(f'mean : {torch.mean(out)}')

    if not os.path.exists('./weight_inner_prod/exp'+exp):
        os.makedirs('./weight_inner_prod/exp'+exp)

    out = out.detach().cpu().numpy()
    mat_list.append(out)


# print('05')

# face_net = networks.F_net(num_classes=1054)
# ocular_net = networks.O_net(num_classes=1054)
# state_dict = torch.load('./results/exp_05/models/model_70.pth.tar', map_location='cpu')
# face_net.load_state_dict(state_dict['face_state_dict'])
# ocular_net.load_state_dict(state_dict['state_dict'])
# face_net.eval()
# ocular_net.eval()
# face_mat = face_net.fc.weight
# ocular_mat = ocular_net.fc.weight

# out = torch.abs(torch.matmul(face_mat, face_mat.t()) - torch.matmul(ocular_mat, ocular_mat.t())) / 2.0

# print('test')
# print(f'mean : {torch.mean(out)}')

# if not os.path.exists('./weight_inner_prod/exp'+exp):
#     os.makedirs('./weight_inner_prod/exp'+exp)

# out = out.detach().cpu().numpy()
# plt.imshow(out, cmap='viridis')
# plt.colorbar()
# plt.ylabel('labels')
# plt.xlabel('labels')
# plt.savefig('./weight_inner_prod/exp'+exp+'/test.jpg')
# plt.close()


print('vanilla')
face_net = networks.F_net(num_classes=1054)
ocular_net = networks.O_net(num_classes=1054)
face_state_dict = torch.load('./results/exp_07/models/model_70.pth.tar', map_location='cpu')
ocular_state_dict = torch.load('./results/exp_06/models/model_70.pth.tar', map_location='cpu')
face_net.load_state_dict(face_state_dict['state_dict'])
ocular_net.load_state_dict(ocular_state_dict['state_dict'])
face_net.eval()
ocular_net.eval()

face_mat = face_net.fc.weight
ocular_mat = ocular_net.fc.weight

out = torch.abs(torch.matmul(face_mat, face_mat.t()) - torch.matmul(ocular_mat, ocular_mat.t())) / 2.0

print('test')
print(f'mean : {torch.mean(out)}')

if not os.path.exists('./weight_inner_prod/vanilla'):
    os.makedirs('./weight_inner_prod/vanilla')

out = out.detach().cpu().numpy()
mat_list.append(out)


mat_list = np.array(mat_list)

np.save('./weight_inner_prod/out_mat.npy', mat_list)

min_val = 0
max_val = mat_list[0].max()

print(min_val)
print(max_val)

mat_list = (mat_list - min_val) / (max_val-min_val)
mat_list = mat_list.clip(max=1.0)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

mat = mat_list[0]
im = ax1.imshow(mat, cmap='afmhot')
# plt.colorbar()
# plt.savefig('./weight_inner_prod/exp01/mat.jpg')
# plt.close()

mat = mat_list[1]
im = ax2.imshow(mat, cmap='afmhot')
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis='both', which='both', length=0)
# plt.colorbar()
# plt.savefig('./weight_inner_prod/exp04/mat.jpg')
# plt.close()


mat = mat_list[2]
im = ax3.imshow(mat, cmap='afmhot')
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.tick_params(axis='both', which='both', length=0)
plt.colorbar(im, ax=[ax1, ax2, ax3])
plt.savefig('./weight_inner_prod/vanilla/mat.jpg')
plt.close()


