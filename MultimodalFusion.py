#!/usr/bin/env Python
# coding=utf-8

from __future__ import print_function
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math

from custom_data_io import custom_dset
from torch.utils.data import DataLoader
import Models as models
from torch.utils import model_zoo

import radam

from sklearn import metrics

import visdom
import numpy as np

import logging
import time

from objectives import cca_loss

np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(levelname)s-%(message)s')
# logging.disable(level=logging.CRITICAL)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# print(os.getcwd())
log_path = os.getcwd() + '/Logs/'
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)

logger.addHandler(fh)

# To use this:
# python -m visdom.server
# http://localhost:8097/
vis = visdom.Visdom(env=u'jj_multimodal_fusion')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 256
epochs = 1
lr = 1e-5
momentum = 0.9
no_cuda = False
seed = 5

log_interval = 30
log_interval_test = 9

l2_decay = 1e-3
root_path = "./"

# jj multimedia
source_list = './jj/jj_train_25c_s.csv'
target_list = './jj/jj_val_25c_s.csv'
validation_list = './jj/jj_val_25c_s.csv'

source_name = 'shallow zone'  # todo
target_name = 'deep zone'
test_name = 'deep zone/validation'

ckpt_path = './ckpt_mm_jj_/'
ckpt_model = './ckpt_mm_jj_trans_221228_7/model_epoch7.pth'
ckpt_model_mlp = './ckpt_mm_jj_221227/model_epoch_mlp80.pth'

parallel = True
blending = True
correlation = True

mlp_pre = False
branch_fixed = False
resume = False

outdim_size = 10
device = torch.device('cuda')
# Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu device type

# Create parent path if it doesn't exist
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True

source_dataset = custom_dset(txt_path=source_list, nx=227, nz=227, labeled=True)
validation_dataset = custom_dset(txt_path=validation_list, nx=227, nz=227, labeled=True)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                           drop_last=True)
target_test_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1,  # todo:
                                pin_memory=True, drop_last=True)

len_source_dataset = len(source_dataset)
len_test_dataset = len(validation_dataset)
len_source_loader = len(source_loader)
len_test_loader = len(target_test_loader)

with open(source_list, 'r') as f:
    lines = f.readlines()
len_source_ones = 0
len_source_zeros = 0
for line in lines:
    items = line.split(',')

if int(items[1]) == 1:
    len_source_ones += 1
len_source_zeros = len_source_dataset - len_source_ones

with open(validation_list, 'r') as f:
    lines = f.readlines()
len_val_ones = 0
len_val_zeros = 0
for line in lines:
    items = line.split(',')

if int(items[1]) == 1:
    len_val_ones += 1
len_val_zeros = len_test_dataset - len_val_ones


def load_pretrain_alex(model, alexnet_model=True):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    if alexnet_model:

        pretrained_dict = model_zoo.load_url(url)
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            if not 'classifier' in k:
                if not "features.0" in k:
                    if not "cls" in k:
                        model_dict[k] = pretrained_dict[k]
    return model


def train(epoch, model, model_mlp=None, optimizer_arg='adamw', blending=blending, ):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .6f}'.format(LEARNING_RATE))

    if parallel:
        if optimizer_arg == 'Adam':
            optimizer = torch.optim.Adam(  # filter(lambda p: p.requires_grad,
                [  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                    {'params': filter(lambda p: p.requires_grad, model.features.parameters())},
                    # lr=LEARNING_RATE / 10
                    {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L7.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L9.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L10.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L12.parameters(), 'lr': LEARNING_RATE},
                ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)  # todo:momentum=momentum,
            optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)
        if optimizer_arg == 'radam':
            optimizer = radam.RAdam(  # filter(lambda p: p.requires_grad,
                [  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                    {'params': filter(lambda p: p.requires_grad, model.features.parameters())},
                    # lr=LEARNING_RATE / 10
                    {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L7.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L9.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L10.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L12.parameters(), 'lr': LEARNING_RATE},
                ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)  # todo:momentum=momentum,
            optimizer_mlp = radam.RAdam(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)
        if optimizer_arg == 'adamw':
            optimizer = radam.AdamW(  # filter(lambda p: p.requires_grad,
                [  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                    {'params': filter(lambda p: p.requires_grad, model.features.parameters())},
                    # lr=LEARNING_RATE / 10
                    {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L7.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L9.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L10.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.L12.parameters(), 'lr': LEARNING_RATE},
                ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)  # todo:momentum=momentum,
            optimizer_mlp = radam.AdamW(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)

    else:
        if optimizer_arg == 'Adam':
            optimizer = torch.optim.Adam(  # filter(lambda p: p.requires_grad,
                [  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                    {'params': filter(lambda p: p.requires_grad, model.features.parameters())},
                    # lr=LEARNING_RATE / 10
                    {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
                    {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
                ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)  # todo:momentum=momentum,
            # optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)
        elif optimizer_arg == 'radam':
            optimizer = radam.RAdam(params=[  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters())},  # lr=LEARNING_RATE / 10
                {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)
            # optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)

        elif optimizer_arg == 'adamw':
            optimizer = radam.AdamW(params=[  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters())},  # lr=LEARNING_RATE / 10
                {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)
            # optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=LEARNING_RATE, weight_decay=l2_decay)

    model.train()
    if parallel or mlp_pre:
        model_mlp.train()

    iter_source = iter(source_loader)
    num_iter = len_source_loader
    TP, TN, FN, FP = 0, 0, 0, 0

    ce_accum = 0.
    for i in range(1, num_iter):
        # if i == 2:
        #     print(i)
        logging.debug('Start %dth iteration...' % (i))
        print('Start %dth iteration...' % (i))
        logging.debug('data_source, label_source = next(iter_source)')
        data_source, label_source, coordinate_source, fluid_source = next(iter_source)
        logging.debug('if i % len_target_loader == 0:')
        if parallel:
            if i % len_source_loader == 0:
                iter_source = iter(source_loader)

        logging.debug('if cuda:')
        if cuda:
            logging.debug('push source data to gpu')
            torch.cuda.empty_cache()
            data_source, label_source, coordinate_source, fluid_source = data_source.cuda(), label_source.cuda(), coordinate_source.cuda(), fluid_source.cuda()
        logging.debug('Variable source data')
        data_source, label_source, coordinate_source, fluid_source = Variable(data_source.float()), Variable(
            label_source.long()), Variable(coordinate_source.float()), Variable(fluid_source.float())

        logging.debug('clear old gradients from the last step')
        optimizer.zero_grad()
        if parallel or mlp_pre:
            optimizer_mlp.zero_grad()
            fluid_feature = model_mlp.features(fluid_source)
        else:
            fluid_feature = 0.

        score_source_pred, _, new_feature_pred, negative_corr = model(data_source,
                                                                      fluid_source,
                                                                      blending=blending,
                                                                      parallel=parallel,
                                                                      fluid_feature=fluid_feature,
                                                                      correlation=correlation,
                                                                      outdim_size=outdim_size,
                                                                      device=device)
        if mlp_pre:
            fluid_pred = model_mlp.classifier(fluid_feature)
        logging.debug('Calculating loss...')

        # softmax_score = F.softmax(score_source_pred, dim=1)  # todo: logistic

        if parallel or blending:
            # print(new_feature_pred)
            prob_source_pred = F.softmax(new_feature_pred, dim=1)
            y_1 = prob_source_pred[:, 1]
            y_0 = prob_source_pred[:, 0]
            ratio_source = y_1 / y_0 * len_source_zeros / len_source_ones
            prob_source_pred_new = torch.Tensor(prob_source_pred.size()).cuda()
            prob_source_pred_new[:, 1] = ratio_source.data / (ratio_source.data + 1)
            prob_source_pred_new[:, 0] = 1 / (ratio_source.data + 1)
            prob_source_pred_new = Variable(prob_source_pred_new)
            # softmax_new_feature = F.softmax(new_feature_pred, dim=1).cuda()
            # blending_softmax = 0.6 * softmax_score + 0.4 * softmax_new_feature
            # blending_log_softmax = torch.log(blending_softmax)
            # loss_cls = F.nll_loss(F.log_softmax(blending_log_softmax, dim=1),
            #                       target=label_source)
            blending_softmax = F.log_softmax(new_feature_pred, dim=1)
            # print('blending_softmax:')
            # print(blending_softmax)
            loss_cls = F.nll_loss(F.log_softmax(new_feature_pred, dim=1),
                                  target=label_source)
        elif mlp_pre == False:
            prob_source_pred = F.softmax(score_source_pred, dim=1)
            y_1 = prob_source_pred[:, 1]
            y_0 = prob_source_pred[:, 0]
            ratio_source = y_1 / y_0 * len_source_zeros / len_source_ones
            prob_source_pred_new = torch.Tensor(prob_source_pred.size()).cuda()
            prob_source_pred_new[:, 1] = ratio_source.data / (ratio_source.data + 1)
            prob_source_pred_new[:, 0] = 1 / (ratio_source.data + 1)
            prob_source_pred_new = Variable(prob_source_pred_new)
            loss_cls = F.nll_loss(F.log_softmax(score_source_pred, dim=1),
                                  target=label_source)  # the negative log likelihood loss
            logging.debug('loss_cls = %s' % (loss_cls))
        else:
            prob_source_pred = F.softmax(fluid_pred, dim=1)
            y_1 = prob_source_pred[:, 1]
            y_0 = prob_source_pred[:, 0]
            ratio_source = y_1 / y_0 * len_source_zeros / len_source_ones
            prob_source_pred_new = torch.Tensor(prob_source_pred.size()).cuda()
            prob_source_pred_new[:, 1] = ratio_source.data / (ratio_source.data + 1)
            prob_source_pred_new[:, 0] = 1 / (ratio_source.data + 1)
            prob_source_pred_new = Variable(prob_source_pred_new)
            loss_mlp = F.nll_loss(F.log_softmax(prob_source_pred, dim=1),
                                  target=label_source)
        gamma = (2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1) * 1e-2  # lambda in DAN paper#todo:denominator: epochs

        ce_accum += torch.abs(loss_cls)
        logging.debug('ce accum = %s' % (ce_accum))

        if mlp_pre:
            loss_cls = 0.
        loss = loss_cls + gamma * negative_corr
        logging.debug('Calculate total loss')
        # loss = loss_cls

        pred = prob_source_pred_new.data.max(1)[1]

        logging.debug('compute training auc, f1 score and accuracy')

        fpr, tpr, thresholds = metrics.roc_curve(y_true=label_source.data.cpu().numpy(),
                                                 y_score=prob_source_pred.data[:, 1].cpu().numpy(),
                                                 pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        TP += ((pred == 1) & (label_source.data.view_as(pred) == 1)).cpu().sum()
        TN += ((pred == 0) & (label_source.data.view_as(pred) == 0)).cpu().sum()
        FN += ((pred == 0) & (label_source.data.view_as(pred) == 1)).cpu().sum()
        FP += ((pred == 1) & (label_source.data.view_as(pred) == 0)).cpu().sum()
        if (TP + FP) != 0:
            p = TP / (TP + FP)
        else:
            p = 0
        if (TP + FN) != 0:
            r = TP / (TP + FN)
        else:
            r = 0
        if (r + p) != 0:
            F1score = 2 * r * p / (r + p)
        else:
            F1score = 0

        train_acc = (TP + TN) / (TP + TN + FP + FN)

        logging.debug('computing the derivative of the loss w.r.t. the params')
        if mlp_pre == False:
            loss.backward()
            # for name, params in model_mlp.named_parameters():
            #     if name.find('weight') != -1:
            #         torch.nn.utils.clip_grad_norm(params, 5e-3)
            #     if name.find('bias') != -1:
            #         torch.nn.utils.clip_grad_norm(params, 5e-3)
            for name, params in model.named_parameters():
                if name.find('weight') != -1:
                    torch.nn.utils.clip_grad_norm(params, 5e-3)
                if name.find('bias') != -1:
                    torch.nn.utils.clip_grad_norm(params, 5e-3)
        else:
            loss_mlp.backward()

        logging.debug('updating params based on the gradients')
        optimizer.step()
        # optimizer.zero_grad()
        if parallel or mlp_pre:
            optimizer_mlp.step()

        if i % log_interval == 0:
            # opts = dict(xlabel='minibatches',
            #             ylabel='Loss',
            #             title='Training Loss',
            #             legend=['Loss']))
            if mlp_pre == False:
                vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=[loss_cls.cpu().data.numpy()],
                         # np.array([loss_cls]),
                         # .cpu().data.numpy(),
                         win='loss_cls',
                         update='append',
                         opts={'title': 'CNN risk'})
            else:
                vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=loss_mlp.cpu().data.numpy(),
                         win='loss_mlp',
                         update='append',
                         opts={'title': 'mlp risk'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([gamma]), win='gamma',
                     update='append',
                     opts={'title': 'penalty parameter'})

            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([auc_value]),
                     win='training auc',
                     update='append',
                     opts={'title': 'training auc'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([F1score]),
                     win='training F1 score',
                     update='append',
                     opts={'title': 'training F1 score'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([train_acc]),
                     win='train_acc',
                     update='append',
                     opts={'title': 'training accuracy'})
            continue
        torch.cuda.empty_cache()
    ce_accum /= len_source_loader
    logging.debug('ce = %s' % (ce_accum))


def test(epoch, model, model_mlp=None, blending=False):
    model.eval()
    if parallel or mlp_pre:
        model_mlp.eval()

    test_loss = 0
    correct = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    F1score = 0
    auc_test_all = 0.
    ce = 0.

    iter_test = iter(target_test_loader)
    num_iter_test = len_test_loader

    auc_arr = np.empty(0)
    label_arr = np.empty(0)
    pred_arr = np.empty(0)

    for i in range(1, num_iter_test):
        logging.debug('Start %dth testing iteration...' % (i))
        print('Start %dth testing iteration...' % (i))
        data, label, coordinate_target, fluid_target = next(iter_test)  # data_shape: torch.Size([32, 3, 224, 224])
        data, label, fluid_target = data.float(), label.long(), fluid_target.float()
        if cuda:
            data, label, fluid_target = data.cuda(), label.cuda(), fluid_target.cuda()
        data, label, fluid_target = Variable(data, volatile=True), Variable(label, volatile=True), Variable(
            fluid_target)
        if parallel or mlp_pre:
            fluid_feature = model_mlp.features(fluid_target)
        else:
            fluid_feature = 0.

        s_output, _, new_feature_pred, _ = model(data, fluid_target, blending=blending, parallel=parallel,
                                                 fluid_feature=fluid_feature)
        if mlp_pre:
            fluid_pred = model_mlp.classifier(fluid_feature)

        if parallel or blending:
            test_loss += F.nll_loss(F.log_softmax(new_feature_pred, dim=1), label, size_average=False).item()
            # .data[0]  # sum up batch loss
            loss_val = F.nll_loss(F.log_softmax(new_feature_pred, dim=1), label).item()
            # .data[0]
            prob_val_pred = F.softmax(new_feature_pred, dim=1)
            y_1 = prob_val_pred[:, 1]
            y_0 = prob_val_pred[:, 0]
            ratio_val = y_1 / y_0 * len_val_zeros / len_val_ones
            prob_val_pred_new = torch.Tensor(prob_val_pred.size()).cuda()
            prob_val_pred_new[:, 1] = ratio_val.data / (ratio_val.data + 1)
            prob_val_pred_new[:, 0] = 1 / (ratio_val.data + 1)
            prob_val_pred_new = Variable(prob_val_pred_new)


        elif mlp_pre == False:
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), label,
                                    size_average=False).item()  # .data[0]  # sum up batch loss
            loss_val = F.nll_loss(F.log_softmax(s_output, dim=1), label).item()  # .data[0]
            prob_val_pred = F.softmax(s_output, dim=1)
            y_1 = prob_val_pred[:, 1]
            y_0 = prob_val_pred[:, 0]
            ratio_val = y_1 / y_0 * len_val_zeros / len_val_ones
            prob_val_pred_new = torch.Tensor(prob_val_pred.size()).cuda()
            prob_val_pred_new[:, 1] = ratio_val.data / (ratio_val.data + 1)
            prob_val_pred_new[:, 0] = 1 / (ratio_val.data + 1)
            prob_val_pred_new = Variable(prob_val_pred_new)

        else:
            test_loss += F.nll_loss(F.log_softmax(fluid_pred, dim=1), label, size_average=False).data[
                0]  # sum up batch loss
            loss_val = F.nll_loss(F.log_softmax(fluid_pred, dim=1), label).data[0]
            prob_val_pred = F.softmax(fluid_pred, dim=1)
            y_1 = prob_val_pred[:, 1]
            y_0 = prob_val_pred[:, 0]
            ratio_val = y_1 / y_0 * len_val_zeros / len_val_ones
            prob_val_pred_new = torch.Tensor(prob_val_pred.size()).cuda()
            prob_val_pred_new[:, 1] = ratio_val.data / (ratio_val.data + 1)
            prob_val_pred_new[:, 0] = 1 / (ratio_val.data + 1)
            prob_val_pred_new = Variable(prob_val_pred_new)

        logging.debug('ce_accum = %s' % (test_loss))  # cross entropy
        pred = prob_val_pred_new.data.max(1)[
            1]  # get the index of the max log-probability, s_output_shape: torch.Size([32, 31])
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        # fpr, tpr, thresholds = metrics.roc_curve(y_true=label_source.data, y_score=prob_source_pred.data[:, 1],
        #                                          pos_label=1)
        # auc_value = metrics.auc(fpr, tpr)

        label_arr = np.append(label_arr, label.data.cpu().numpy())
        pred_arr = np.append(pred_arr, prob_val_pred.data[:, 1].cpu().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(y_true=label.data.cpu().numpy(),
                                                 y_score=prob_val_pred.data[:, 1].data.cpu().numpy(),
                                                 pos_label=1)

        # if np.isnan(tpr).all():
        #     tpr=np.full(shape=tpr.shape,fill_value=1e-8)

        auc_value_test = metrics.auc(fpr, tpr)  # todo
        auc_test_all += auc_value_test
        TP += ((pred == 1) & (label.data.view_as(pred) == 1)).cpu().sum()
        TN += ((pred == 0) & (label.data.view_as(pred) == 0)).cpu().sum()
        FN += ((pred == 0) & (label.data.view_as(pred) == 1)).cpu().sum()
        FP += ((pred == 1) & (label.data.view_as(pred) == 0)).cpu().sum()

        if (TP + FP) != 0:
            p = TP / (TP + FP)
        else:
            p = 0
        if (TP + FN) != 0:
            r = TP / (TP + FN)
        else:
            r = 0
        if (r + p) != 0:
            F1score = 2 * r * p / (r + p)
        else:
            F1score = 0
        if i % log_interval_test == 0:
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([auc_value_test]),
                     win='testing auc',
                     update='append',
                     opts={'title': 'testing auc'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([F1score]),
                     win='testing F1 score',
                     update='append',
                     opts={'title': 'testing F1 score'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([correct / len_test_dataset]),
                     win='test_acc',
                     update='append',
                     opts={'title': 'testing accuracy'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([loss_val]),
                     win='loss_val',
                     update='append',
                     opts={'title': 'testing loss'})

    fpr, tpr, thresholds = metrics.roc_curve(y_true=label_arr,
                                             y_score=pred_arr,
                                             pos_label=1)
    auc_value_test = metrics.auc(fpr, tpr)
    vis.line(X=np.array([epoch]), Y=np.array([auc_value_test]),
             win='testing auc per epoch',
             update='append',
             opts={'title': 'testing auc per epoch'})

    test_loss /= len_test_dataset
    logging.debug('ce = %s' % (test_loss))  # cross entropy
    auc_test_all /= num_iter_test
    print('\n{}  {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1score: {}, auc: {}\n'.format(
        datetime.now(), test_name, test_loss, correct, len_test_dataset,
        100. * correct / len_test_dataset, F1score, auc_test_all))
    logging.debug('auc = %s' % (auc_test_all))
    torch.cuda.empty_cache()

    return correct


def load_ckpt(model):
    ckpt_dict = torch.load(ckpt_model)
    new_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt_dict.items() if k in new_dict}
    new_dict.update(pretrained_dict)
    model.load_state_dict(new_dict)

    return model


def load_ckpt_mlp(model_mlp):
    model_mlp.load_state_dict(torch.load(ckpt_model_mlp))
    return model_mlp


if __name__ == '__main__':
    model = models.DAN_with_Alex(num_classes=2, branch_fixed=branch_fixed,
                                 transfer=transfer)  # , correlation=correlation)
    print(model)
    if parallel or mlp_pre:
        model_mlp = models.new_net()
        # model_mlp.classifier = torch.nn.Sequential(torch.nn.Linear(256, 2))
        print(model_mlp)
    else:
        model_mlp = None

    correct = 0

    if cuda:
        model.cuda()
        if parallel or mlp_pre:
            model_mlp.cuda()
    if mlp_pre == False:
        model = load_pretrain_alex(model, alexnet_model=True)
    if resume:
        model = load_ckpt(model)
        if parallel:
            model_mlp = load_ckpt_mlp(model_mlp)
    if parallel:
        model = load_ckpt(model)
    for epoch in range(1, epochs + 1):
        train(epoch, model, model_mlp=model_mlp, blending=blending, )
        # t_correct = 0.
        t_correct = test(epoch, model, model_mlp=model_mlp, blending=blending)
        # Save models.
        if mlp_pre == False:
            ckpt_name = os.path.join(ckpt_path, 'model_epoch' + str(epoch) + '.pth')
            print('Save model: {}'.format(ckpt_name))
            torch.save(obj=model.state_dict(), f=ckpt_name)
        if parallel or mlp_pre:
            ckpt_name_mlp = os.path.join(ckpt_path, 'model_epoch_mlp' + str(epoch) + '.pth')
            print('Save model: {}'.format(ckpt_name_mlp))
            torch.save(obj=model_mlp.state_dict(), f=ckpt_name_mlp)
        if t_correct > correct:
            correct = t_correct
        current_acc = 100. * t_correct / len_test_dataset
        print('{} source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            datetime.now(), source_name, target_name, correct, 100. * correct / len_test_dataset))
