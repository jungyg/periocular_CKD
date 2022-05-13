from __future__ import print_function

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def makedir(dir_name):
    if os.path.isdir(dir_name):
        pass
    else:
        os.makedirs(dir_name)

def model_exists(dir):
    file_list = os.listdir(dir)
    for f in file_list:
        if 'model' in f:
            return True
    return False

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterTensor(object):
    """Computes and stores the average and current value"""
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.avg = torch.zeros(self.size).cuda()
        self.sum = torch.zeros(self.size).cuda()
        self.count = 0

    def update(self, val):
        n = val.shape[0]
        self.sum += torch.sum(val, dim=0)
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def imsave(img, fname):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig(fname, bbox_inches='tight')

def calculate_cmc_inner(gallery_embedding, probe_embedding, gallery_label, probe_label, last_rank=20):
    """
    :param gallery_embedding: [num of gallery images x embedding size] (n x e) torch float tensor
    :param probe_embedding: [num of probe images x embedding size] (m x e) torch float tensor
    :param gallery_label: [num of gallery images x num of labels] (n x l) torch one hot matrix
    :param label: [num of probe images x num of labels] (m x l) torch one hot matrix
    :param last_rank: the last rank of cmc curve
    :return: (x_range, cmc) where x_range is range of ranks and cmc is probability list with length of last_rank
    """
    nof_query = probe_label.shape[0]
    prediction_score = torch.matmul(probe_embedding, gallery_embedding.t())
    gt_similarity = torch.matmul(probe_label, gallery_label.t())
    _, sorted_similarity_idx = torch.sort(prediction_score, dim=1, descending=True)
    cmc = torch.zeros(last_rank).type(torch.float32)
    for i in range(nof_query):
        gt_vector = (gt_similarity[i] > 0).type(torch.float32)
        pred_idx = sorted_similarity_idx[i]
        predicted_gt = gt_vector[pred_idx]
        first_gt = torch.nonzero(predicted_gt).type(torch.int)[0]
        if first_gt < last_rank:
            cmc[first_gt:] += 1
    cmc /= nof_query

    if cmc.device.type == 'cuda':
        cmc = cmc.cpu()

    x_range = np.arange(0,last_rank)+1

    return x_range, cmc.numpy()

def calculate_cmc(gallery_embedding, probe_embedding, gallery_label, probe_label, last_rank=20):
    """
    :param gallery_embedding: [num of gallery images x embedding size] (n x e) torch float tensor
    :param probe_embedding: [num of probe images x embedding size] (m x e) torch float tensor
    :param gallery_label: [num of gallery images x num of labels] (n x l) torch one hot matrix
    :param label: [num of probe images x num of labels] (m x l) torch one hot matrix
    :param last_rank: the last rank of cmc curve
    :return: (x_range, cmc) where x_range is range of ranks and cmc is probability list with length of last_rank
    """
    gallery_embedding = gallery_embedding.type(torch.float32)
    probe_embedding = probe_embedding.type(torch.float32)
    gallery_label = gallery_label.type(torch.float32)
    probe_label = probe_label.type(torch.float32)


    nof_query = probe_label.shape[0]
    gallery_embedding /= torch.norm(gallery_embedding, p=2, dim=1, keepdim=True)
    probe_embedding /= torch.norm(probe_embedding, p=2, dim=1, keepdim=True)
    prediction_score = torch.matmul(probe_embedding, gallery_embedding.t())
    gt_similarity = torch.matmul(probe_label, gallery_label.t())
    _, sorted_similarity_idx = torch.sort(prediction_score, dim=1, descending=True)
    cmc = torch.zeros(last_rank).type(torch.float32)
    for i in range(nof_query):
        gt_vector = (gt_similarity[i] > 0).type(torch.float32)
        pred_idx = sorted_similarity_idx[i]
        predicted_gt = gt_vector[pred_idx]
        first_gt = torch.nonzero(predicted_gt).type(torch.int)[0]
        if first_gt < last_rank:
            cmc[first_gt:] += 1
    cmc /= nof_query

    if cmc.device.type == 'cuda':
        cmc = cmc.cpu()

    x_range = np.arange(0,last_rank)+1

    return x_range, cmc.numpy()

def calculate_roc(thresholds, embeddings1, embeddings2, similarity, metric='cos', nof_folds=10):
    """
    :param thresholds: list of thresholds (e.g. [0, 0.1, 0.2 ... 0.9, 1]
    :param embeddings1: [num of images x embedding vector size ] matrix of first image for verification
    :param embeddings2: [num of images x embedding ector size ] matrix of second image for verification
    :param similarity: [num of images] size vector indicating whether embeddings1 and 2 have same identity or not
    :param metric: default is 'cos' which is cosine distance, or can use l2 distance when input 'l2'
    :param nof_folds: number of folds for cross validation. default value = 10
    :return: [num of thresholds] size vector tpr, fpr
             [num of fold] size vector accuracy, and best thresholds
    """
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nof_pairs = min(len(embeddings1.shape[0]), len(similarity))
    nof_thresholds = len(thresholds)
    kfold = KFold(n_splits=nof_folds, shuffle=False)

    tprs = np.zeros((nof_folds, nof_thresholds))
    fprs = np.zeros((nof_folds, nof_thresholds))
    accuracy = np.zeros((nof_folds))
    best_thresholds = np.zeros((nof_folds))
    indices = np.arange(nof_pairs)

    if metric == 'l2':
        diff = embeddings1 - embeddings2
        dist = np.sum(np.square(diff), 1)
    else:
        dist = np.matmul(embeddings1, embeddings2.transpose(0,1))


    for fold_idx, (train, test) in enumerate(kfold.split(indices)):
        acc_test = np.zeros((nof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_test[threshold_idx] = calculate_accuracy(threshold, dist[train], similarity[train])
        best_threshold_idx = np.argmax(acc_test)
        best_thresholds[fold_idx] = thresholds[best_threshold_idx]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test], similarity[test])

        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_idx], dist[test], similarity[test])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, similarity):
    predict = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict, similarity))
    fp = np.sum(np.logical_and(predict, np.logical_not(similarity)))
    tn = np.sum(np.logical_and(np.logical_not(predict), np.logical_not(similarity)))
    fn = np.sum(np.logical_and(np.logical_not(predict), similarity))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


if __name__ == '__main__':

    pass
