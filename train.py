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
from models import *


from importlib import import_module
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset import face_ocular

from helper.util import *
from helper.loops import train_vanilla, train_distill, validate_vanilla, validate_distill
import socket
import argparse
from config import config as cfg




def train():
        
    cfg.update_with_yaml("ckd.yaml")
    cfg.freeze()

    torch.backends.cudnn.enabled = True
    # torch.cuda.set_device(cfg.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)


    root_dir = os.path.join(cfg.result_path, f'DISTILL-{cfg.distill}-NET-{cfg.network}-BN-{cfg.batchnorm}-DSET-{cfg.dataset}')
    model_dir = os.path.join(root_dir, 'models')
    log_dir = os.path.join(root_dir, 'log')

    makedir(root_dir)
    makedir(model_dir)
    makedir(log_dir)


    log_dict = {'train_face_ce_loss': [], 'train_ocular_ce_loss':[], 'train_ocular_kl_loss':[], 'train_face_kl_loss':[], 
                'train_total_loss':[], 'train_face_acc': [], 'train_ocular_acc':[], 'train_acc': [],
                'val_face_ce_loss': [], 'val_ocular_ce_loss':[], 'val_face_kl_loss':[],  'val_ocular_kl_loss':[], 'val_acc' : [],
                'val_total_loss': [], 'val_face_acc':[], 'val_ocular_acc':[], 'epoch':[], 'best_acc':-1, 'best_acc_epoch':0}


    print('>>>> loading training dataset')
    trainset = face_ocular.train_dataset(dset_type='train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    num_trainset = len(trainset)
    num_classes = trainset.num_classes

    valset = face_ocular.train_dataset(dset_type='val')
    val_loader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, num_workers=4)
    num_valset = len(valset)


    print('{} training images loaded'.format(num_trainset))
    print('{} validating images loaded'.format(num_valset))
    print('{} training identities loaded'.format(num_classes))
    print('')

    print('>>>> loading module')
    module = import_module('models.'+cfg.network)
    model = getattr(module, cfg.network)().cuda()


    ce_crit = nn.CrossEntropyLoss().cuda()
    kl_crit = nn.KLDivLoss(reduction='batchmean').cuda()
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


    current_epoch = 1
    ## load checkpoint if exists
    if os.path.exists(os.path.join(model_dir, 'last_checkpoint.pth')):
        state_dict = torch.load(os.path.join(model_dir, 'last_checkpoint.pth'))
        log_dict = state_dict['log_dict']
        model.load_state_dict(state_dict['state_dict'])
        optim.load_state_dict(state_dict['optimizer'])
        current_epoch = state_dict['epoch']
        print('>>>> Resume training from epoch : %d !!!' % (current_epoch))

    print('>>>> start training')
    for epoch in range(current_epoch, cfg.epochs+1):
        
        ### train loop
        if cfg.network == 'face_network' or cfg.network == 'ocular_network':
            acc, losses = train_vanilla(epoch, train_loader, model, ce_crit, optim, cfg)
            log_dict['train_total_loss'].append(losses)
            log_dict['train_acc'].append(acc)

        else:
            acc, loss_list = train_distill(epoch, train_loader, model, ce_crit, kl_crit, optim, cfg)
            face_ce_loss, face_kl_loss, ocular_ce_loss, ocular_kl_loss, loss = loss_list
            log_dict['train_face_ce_loss'].append(face_ce_loss)
            log_dict['train_face_kl_loss'].append(face_kl_loss)
            log_dict['train_ocular_ce_loss'].append(ocular_ce_loss)
            log_dict['train_ocular_kl_loss'].append(ocular_kl_loss)
            log_dict['train_total_loss'].append(loss)
            log_dict['train_ocular_acc'].append(acc)


        ### validate loop
        if cfg.network == 'face_network' or cfg.network == 'ocular_network':
            acc, losses = validate_vanilla(val_loader, model, ce_crit, cfg)
            log_dict['val_total_loss'].append(losses)
            log_dict['val_acc'].append(acc)

        else:
            acc, loss_list = validate_distill(val_loader, model, ce_crit, kl_crit, cfg)
            face_ce_loss, face_kl_loss, ocular_ce_loss, ocular_kl_loss, loss = loss_list
            log_dict['val_face_ce_loss'].append(face_ce_loss)
            log_dict['val_face_kl_loss'].append(face_kl_loss)
            log_dict['val_ocular_ce_loss'].append(ocular_ce_loss)
            log_dict['val_ocular_kl_loss'].append(ocular_kl_loss)
            log_dict['val_total_loss'].append(loss)
            log_dict['val_ocular_acc'].append(acc)
            log_dict['epoch'].append(epoch)
            val_ocular_acc = acc

        if epoch in cfg.decay_epochs:
            for params in optim.param_groups:
                params['lr'] /= 10.0

        if val_ocular_acc > log_dict['best_acc']:
            log_dict['best_acc'] = val_ocular_acc
            log_dict['best_acc_epoch'] = epoch
            state_dict = {'epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optim.state_dict(),
                'log_dict':log_dict
                }
            torch.save(state_dict, os.path.join(model_dir, f'best_checkpoint.pth'))


        with open(os.path.join(log_dir, 'last_log_json.json'), 'w') as f:
            json.dump(log_dict, f, indent=2)

        state_dict = {
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'optimizer':optim.state_dict(),
            'log_dict':log_dict
        }        
        torch.save(state_dict, os.path.join(model_dir, 'last_checkpoint.pth'))


    # for k,v in log_dict.items():
    #     if 'train' in k:
    #         x_range = range(len(log_dict[k]))
    #         plt.plot(x_range, log_dict[k], label=k)
    #         plt.legend(loc='upper right')
    #         plt.savefig(os.path.join(figure_dir, k + '.jpg'))
    #         plt.close()
    #     elif 'val' in k:
    #         plt.plot(val_list, log_dict[k], label=k)
    #         plt.legend(loc='upper right')
    #         plt.savefig(os.path.join(figure_dir, k + '.jpg'))
    #         plt.close()


    # with open(os.path.join(figure_dir, 'hyperparam_json.json'), 'w') as f:
    #     json.dump(param_dict, f, indent=2)



if __name__ == "__main__":
    train()