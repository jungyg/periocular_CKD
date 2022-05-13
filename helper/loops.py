from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from .util import AverageMeter, accuracy, AverageMeterTensor
import sys

def train_vanilla(epoch, train_loader, model, ce_crit, optim, cfg):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    for i, (face, ocular, label, onehot) in enumerate(train_loader):

        num_img, _, _, _ = face.size()
        label = label.cuda()

        # ===================forward=====================
        if cfg.network == 'face_network':
            face = face.cuda()
            _, output = model(face)
            face_ce_loss = ce_crit(output, label)
            loss = face_ce_loss 

        elif cfg.network == 'ocular_network':
            ocular = ocular.cuda()
            _, output = model(ocular)
            ocular_ce_loss = ce_crit(output, label)
            loss = ocular_ce_loss 



        # ===================backward=====================
        optim.zero_grad()
        loss.backward()
        optim.step()

        # ===================meters=====================

        acc1  = accuracy(output, label, topk=(1,))
        losses.update(loss.item(), num_img)
        acc.update(acc1[0].item(), num_img)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if i % 100 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t')
            sys.stdout.flush()

    print(f'[[Train]] * Acc {acc.avg:.3f} loss {loss.avg:.3f}')
    return acc.avg, losses.avg


def train_distill(epoch, train_loader, model, ce_crit, kl_crit, optim, cfg):
    model.train()

    batch_time = AverageMeter()
    face_ce_losses = AverageMeter()
    face_kl_losses = AverageMeter()
    ocular_ce_losses = AverageMeter()
    ocular_kl_losses = AverageMeter()
    total_losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    for i, (face, ocular, label, onehot) in enumerate(train_loader):

        num_img, _, _, _ = face.size()
        face = face.cuda()
        ocular = ocular.cuda()
        label = label.cuda()

        # ===================forward=====================


        if cfg.network == 'shared_ocular_network':
            _, _, ocular1_out, ocular_out = model(ocular)
            face_ce_loss = ce_crit(ocular1_out, label)
            face_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular1_out/cfg.tau, dim=1), torch.softmax(ocular_out.detach()/cfg.tau, dim=1))
            ocular_ce_loss = ce_crit(ocular_out, label)
            ocular_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular_out/cfg.tau, dim=1), torch.softmax(ocular1_out.detach()/cfg.tau, dim=1))
            loss = face_ce_loss + ocular_ce_loss + face_kl_loss + ocular_kl_loss

        else:
            _, _, face_out, ocular_out = model(face, ocular)
            face, ocular = face.cuda(), ocular.cuda()

            face_kl_loss = 0
            ocular_kl_loss = 0
            face_ce_loss = ce_crit(face_out, label)
            ocular_ce_loss = ce_crit(ocular_out, label)
            if cfg.ocular_distill:
                face_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(face_out/cfg.tau, dim=1), torch.softmax(ocular_out.detach()/cfg.tau, dim=1))
            if cfg.face_distill:
                ocular_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular_out/cfg.tau, dim=1), torch.softmax(face_out.detach()/cfg.tau, dim=1))

            loss = face_ce_loss + ocular_ce_loss + face_kl_loss + ocular_kl_loss

        acc1  = accuracy(ocular_out, label, topk=(1,))



        # ===================backward=====================
        optim.zero_grad()
        loss.backward()
        optim.step()

        # ===================meters=====================
        face_ce_losses.update(face_ce_loss.item(), num_img)
        face_kl_losses.update(face_kl_loss.item(), num_img)
        ocular_ce_losses.update(ocular_ce_loss.item(), num_img)
        ocular_kl_losses.update(ocular_kl_loss.item(), num_img)
        total_losses.update(loss.item(), num_img)
        acc.update(acc1[0].item(), num_img)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
    print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Face_CE {face_ce_losses.val:.4f} ({face_ce_losses.avg:.4f})\t'
            f'Face_KL {face_kl_losses.val:.4f} ({face_kl_losses.avg:.4f})\t'
            f'Ocular_CE {ocular_ce_losses.val:.4f} ({ocular_ce_losses.avg:.4f})\t'
            f'Ocular_KL {ocular_kl_losses.val:.4f} ({ocular_kl_losses.avg:.4f})\t'
            f'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})\t'
            f'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t')
    sys.stdout.flush()

    # print(f'[[Train]] * Acc {acc.avg:.3f} \t'
    #         f'Face_CE {face_ce_losses.avg:.4f}\t'
    #         f'Face_KL {face_kl_losses.avg:.4f}\t'
    #         f'Ocular_CE {ocular_ce_losses.avg:.4f}\t'
    #         f'Ocular_KL {ocular_kl_losses.avg:.4f}\t'
    #         f'Loss {total_losses.avg:.4f}\t')

    
    return acc.avg, [face_ce_losses.avg, face_kl_losses.avg, ocular_ce_losses.avg, ocular_kl_losses.avg, total_losses.avg]



def validate_vanilla(val_loader, model, ce_crit, cfg):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (face, ocular, label, onehot) in enumerate(val_loader):
            num_img, _, _, _ = face.size()
            label = label.cuda()
            # ===================forward=====================
            if cfg.network == 'face_network':
                face = face.cuda()
                _, output = model(face)
                face_ce_loss = ce_crit(output, label)
                loss = face_ce_loss 

            elif cfg.network == 'ocular_network':
                ocular = ocular.cuda()
                _, output = model(ocular)
                ocular_ce_loss = ce_crit(output, label)
                loss = ocular_ce_loss 




            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
            acc1  = accuracy(output, label, topk=(1,))
            losses.update(loss.item(), num_img)
            acc.update(acc1[0].item(), num_img)

            # print info
            if i % 100 == 0:
                print(f'Epoch: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t')
                sys.stdout.flush()

        print(f'[[Val]] * Acc {acc.avg:.3f} loss {losses.avg:.3f}')
    return acc.avg, losses.avg


def validate_distill(val_loader, model, ce_crit, kl_crit, cfg):
    """validation"""
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        batch_time = AverageMeter()
        face_ce_losses = AverageMeter()
        face_kl_losses = AverageMeter()
        ocular_ce_losses = AverageMeter()
        ocular_kl_losses = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        end = time.time()
        for i, (face, ocular, label, onehot) in enumerate(val_loader):
            num_img, _, _, _ = face.size()
            face = face.cuda()
            ocular = ocular.cuda()
            label = label.cuda()

            # ===================forward=====================
            if cfg.network == 'shared_ocular_network':
                _, _, ocular1_out, ocular_out = model(ocular)
                face_ce_loss = ce_crit(ocular1_out, label)
                face_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular1_out/cfg.tau, dim=1), torch.softmax(ocular_out.detach()/cfg.tau, dim=1))
                ocular_ce_loss = ce_crit(ocular_out, label)
                ocular_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular_out/cfg.tau, dim=1), torch.softmax(ocular1_out.detach()/cfg.tau, dim=1))
                loss = face_ce_loss + ocular_ce_loss + face_kl_loss + ocular_kl_loss

            else:
                _, _, face_out, ocular_out = model(face, ocular)
                face, ocular = face.cuda(), ocular.cuda()

                face_kl_loss = 0
                ocular_kl_loss = 0
                face_ce_loss = ce_crit(face_out, label)
                ocular_ce_loss = ce_crit(ocular_out, label)
                if cfg.ocular_distill:
                    face_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(face_out/cfg.tau, dim=1), torch.softmax(ocular_out.detach()/cfg.tau, dim=1))
                if cfg.face_distill:
                    ocular_kl_loss = cfg.tau * cfg.tau * kl_crit(torch.log_softmax(ocular_out/cfg.tau, dim=1), torch.softmax(face_out.detach()/cfg.tau, dim=1))

                loss = face_ce_loss + ocular_ce_loss + face_kl_loss + ocular_kl_loss

            acc1 = accuracy(ocular_out, label, topk=(1,))



            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), num_img)
            face_ce_losses.update(face_ce_loss.item(), num_img)
            face_kl_losses.update(face_kl_loss.item(), num_img)
            ocular_ce_losses.update(ocular_ce_loss.item(), num_img)
            ocular_kl_losses.update(ocular_kl_loss.item(), num_img)
            acc.update(acc1[0].item(), num_img)


            # print info
        print(f'Epoch: [{i}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Face_CE {face_ce_losses.val:.4f} ({face_ce_losses.avg:.4f})\t'
                f'Face_KL {face_kl_losses.val:.4f} ({face_kl_losses.avg:.4f})\t'
                f'Ocular_CE {ocular_ce_losses.val:.4f} ({ocular_ce_losses.avg:.4f})\t'
                f'Ocular_KL {ocular_kl_losses.val:.4f} ({ocular_kl_losses.avg:.4f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t')
        sys.stdout.flush()
        # print(f'[[Val]] * Acc {acc.avg:.3f} \t'
        #         f'Face_CE {face_ce_losses.avg:.4f}\t'
        #         f'Face_KL {face_kl_losses.avg:.4f}\t'
        #         f'Ocular_CE {ocular_ce_losses.avg:.4f}\t'
        #         f'Ocular_KL {ocular_kl_losses.avg:.4f}\t'
        #         f'Loss {losses.avg:.4f}\t')
        
        return acc.avg, [face_ce_losses.avg, face_kl_losses.avg, ocular_ce_losses.avg, ocular_kl_losses.avg, losses.avg]



