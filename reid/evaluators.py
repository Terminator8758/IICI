from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs, lowlayer=False, use_transformer=False, ret_head_feat=False, cams=''):
    inputs = to_torch(inputs).cuda()
    if use_transformer:
        outputs = model(inputs, lowlayer=lowlayer, use_transformer=True)
    elif ret_head_feat:
        outputs = model(inputs, cams, ret_head_feat=True)
    elif lowlayer:
        outputs = model(inputs, lowlayer=lowlayer)
    else:
        outputs = model(inputs)
    #if use_transformer:
    #    outputs[0] = outputs[0].data.cpu()
    #    outputs[1] = outputs[1].data.cpu()
    #else:
    if not use_transformer:
        outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50, lowlayer=False, use_transformer=False, ret_head_feat=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    features2 = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cams, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, lowlayer, use_transformer, ret_head_feat, cams.cuda())  # new added
            if use_transformer:
                batch_feat1 = outputs[0]
                batch_feat2 = outputs[1]
                for fname, output1, output2, pid in zip(fnames, batch_feat1, batch_feat2, pids):
                    features[fname] = output1
                    features2[fname] = output2
                    labels[fname] = pid
            else:
                for fname, output, pid in zip(fnames, outputs, pids):
                    #print('output len= {}, output[0] shape= {}, output[1] shape= {}'.format(len(output), output[0].shape, output[1].shape))
                    features[fname] = output
                    labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
    if use_transformer:
        return features, labels, features2
    else:
        return features, labels


def extract_features_temp(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cams, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            imgs = to_torch(imgs).cuda()
            outputs = model(imgs)
            for fname, output in zip(fnames, outputs):
                features[fname] = output

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
    return features


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query)
        distmat_gg = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
