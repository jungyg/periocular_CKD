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
from collections import OrderedDict
import matplotlib.pyplot as plt
import dload
from torch.utils.tensorboard import SummaryWriter
from utils import *
import json
import sklearn.metrics as metrics
import shared_networks as shared_networks

torch.backends.cudnn.enabled = True
torch.cuda.set_device(0)

exp = '08'
epoch = '90'

root_dir = os.path.join('results', 'exp_'+exp)
model_dir = os.path.join(root_dir, 'models')
figure_dir = os.path.join(root_dir, 'figures', epoch, 'ocular')

makedir(root_dir)
makedir(model_dir)
makedir(figure_dir)

batch_size = 500
embedding_size = 512
nof_train_iden = 1054

model = shared_networks.shared_network(num_classes=1054)
model = shared_networks.shared_network_v2(num_classes=1054)

if exp == '05' or exp == '06':
    model = networks.O_net(num_classes=1054)
if exp == '07':
    model = networks.F_net(num_classes=1054)
## load model
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_'+epoch+'.pth.tar'), map_location='cpu')['state_dict'])
model = model.cuda()
model.eval()

db_dict = {'val':0, 'ethnic':0, 'pubfig':0, 'facescrub':0,'wiki':0, 'ar': 0, 'ytf':0}

### trainset
print('>>>>>>>>>>>>>>>>    dbi score')
# print('>>>> loading val dataset')
valset = dload.train_dataset(dset_type='val')
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=4)
nof_valset = len(valset)
nof_val_iden = valset.nof_identity

embedding_mat = torch.zeros((nof_valset, embedding_size)).cuda()
label_list = torch.zeros((nof_valset)).cuda()
# print('{} val images loaded'.format(nof_valset))
# print('{} val identities loaded'.format(nof_train_iden))

start = time.time()
with torch.no_grad():
    for i, (face, ocular, label, onehot) in enumerate(val_loader):
        nof_img = face.shape[0]
        ocular = ocular.cuda()
        label = label.cuda()

        feature = model.get_ocular_feature(ocular)

        embedding_mat[i*batch_size:i*batch_size+nof_img, :] = feature.detach().clone()
        label_list[i*batch_size:i*batch_size+nof_img] = label.clone()

    embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)
    embedding_mat = embedding_mat.cpu().numpy()
    label_list = label_list.cpu().numpy()
    db_score = metrics.davies_bouldin_score(embedding_mat, label_list)
    db_dict['val'] = db_score

end = time.time()
# print('time took : {} seconds'.format(end-start))
# print('val set')
print(db_score)
# print()



dset_list = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']


for dset in dset_list:
    if dset == 'ethnic' or dset == 'ytf':
        dset_type = 'probe'
    else:
        dset_type = 'gallery'

    # print('>>>> loading '+dset+' benchmarking dataset')
    benchset = dload.benchmark_dataset(dset_name=dset, dset_type=dset_type)
    bench_loader = torch.utils.data.DataLoader(benchset, batch_size=batch_size, num_workers=4)
    nof_benchset = len(benchset)
    nof_bench_iden = benchset.nof_identity
    embedding_mat = torch.zeros((nof_benchset, embedding_size)).cuda()
    label_list = torch.zeros((nof_benchset)).cuda()
    # print('{} {} images loaded'.format(nof_benchset, dset))
    # print('{} {} identities loaded'.format(nof_bench_iden, dset))
    start = time.time()
    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(bench_loader):
            nof_img = ocular.shape[0]
            ocular = ocular.cuda()
            onehot = onehot.cuda()
            label = label.cuda()

            feature = model.get_ocular_feature(ocular)

            embedding_mat[i * batch_size:i * batch_size + nof_img, :] = feature.detach().clone()
            label_list[i * batch_size:i * batch_size + nof_img] = label.clone()

        embedding_mat /= torch.norm(embedding_mat, p=2, dim=1, keepdim=True)
        embedding_mat = embedding_mat.cpu().numpy()
        label_list = label_list.cpu().numpy()
        db_score = metrics.davies_bouldin_score(embedding_mat, label_list)
        db_dict[dset] = db_score

        end = time.time()
        # print('time took : {} seconds'.format(end - start))
        print(db_score)
        # print('')


with open(os.path.join(figure_dir, 'dbi_dict.pkl'), 'wb') as f:
    pickle.dump(db_dict, f)

with open(os.path.join(figure_dir, 'dbi_json.json'), 'w') as f:
    json.dump(db_dict, f, indent=2)

