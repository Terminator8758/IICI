from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.meters import AverageMeter



class SCT_Trainer(object):
    def __init__(self, encoder, memory, arch='', has_aug_transform=True, detr_model=None):
        super(SCT_Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.arch = arch
        self.has_aug_transform = has_aug_transform
        self.detr_model = detr_model
     

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, class_subcam=''):
        self.encoder.train()

        if self.detr_model is not None:
            self.detr_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_fc = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()

            data_time.update(time.time() - end)

            total_loss = 0
            loss = 0

            # process inputs
            imgs, labels, cams, tgt_labels = self._parse_data(inputs)

            if len(class_subcam) > 0:
                tgt_labels = cams  # replace with original camera label
                cams = class_subcam[labels]  # replace with sub-camera label
            
            if self.has_aug_transform:
                concated_imgs = torch.zeros((imgs.size(0)*2, imgs.size(1), imgs.size(2), imgs.size(3)), dtype=imgs.dtype).cuda()            
                idx1 = torch.arange(0, len(concated_imgs), 2)
                idx2 = torch.arange(1, len(concated_imgs), 2)
                concated_imgs[idx1] = imgs
                concated_imgs[idx2] = inputs[1].cuda()  # jitter imgs
                imgs = concated_imgs
   
            # ============== model forward ================
            if self.arch == 'resnet50_nl':
                features, featmap = self.encoder(imgs)  # the default forward 
            else:
                features = self.encoder(imgs)  # ViT-S forward

            trans_feat = ''

            if self.has_aug_transform:                 
                #print('batch feature shape= {}, dtype= {};  idx1 shape= {}, dtype= {}'.format(features.shape, features.dtype, idx1.shape, idx1.dtype))
                ori_features = features[idx1]
                jitter_features = features[idx2]
                if self.detr_model is not None:
                    trans_feat = featmap[idx1]  
                    trans_feat = self.detr_model(trans_feat)  

                loss += self.memory(ori_features, labels, cams, epoch, batch_ind=i, tgt_labels=tgt_labels,
                                    aug_features=jitter_features, trans_feat=trans_feat) 
            else:
                if self.detr_model is not None:
                    trans_feat = self.detr_model(featmap)

                loss += self.memory(features, labels, cams, epoch, batch_ind=i, tgt_labels=tgt_labels, trans_feat=trans_feat)                                 
                 

            total_loss += loss
 
            optimizer.zero_grad()

            total_loss.backward()
            
            optimizer.step()

            losses.update(total_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        # inputs: (img, fname, pid, camid, img_index) or (img, fname, pid, subcam_id, camid)

        imgs = inputs[0]
        labels = inputs[2]
        cams = inputs[3]
        tgt_labels = inputs[4]

        return imgs.cuda(), labels.cuda(), cams.cuda(), tgt_labels.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


