from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch,time
from torch import nn
import torch.nn.functional as F
import random


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def cosine_dist(x, y):
    # m, n = x.size(0), y.size(0)
    dist = 1 - torch.matmul(x, y.t())
    # dist = torch.ones(dist.size()).float() - dist
    return dist


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def distance_mining(dist_mat, labels, cameras):
    assert len(dist_mat.size())==2
    assert dist_mat.size(0)==dist_mat.size(1)
 
    N=dist_mat.size(0)
    
    is_pos=labels.expand(N,N).eq(labels.expand(N,N).t())# & cameras.expand(N,N).eq(cameras.expand(N,N).t())
    # is_neg=labels.expand(N,N).ne(labels.expand(N,N).t()) # | cameras.expand(N,N).ne(cameras.expand(N,N).t())
    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    d1=d1.squeeze(1)
    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2=d1.new().resize_as_(d1).fill_(0)
    d3=d1.new().resize_as_(d1).fill_(0)
    d2ind=[]
    for i in range(N):
        sorted_tensor,sorted_index=torch.sort(dist_mat[i])
        cam_id=cameras[i]
        B,C=False,False
        for ind in sorted_index:
            if labels[ind]==labels[i]:
                continue
            if B==False and cam_id==cameras[ind]:
                d3[i]=dist_mat[i][ind]
                B=True
            if C==False and cam_id!=cameras[ind]:
                d2[i]=dist_mat[i][ind]
                C=True
                d2ind.append(ind)
            if B and C:
                break
    return d1, d2, d3, d2ind


class MCNL_Loss(object):
    """Multi-camera negative loss
        In a mini-batch,
       d1=(A,A'), A' is the hardest true positive. 
       d2=(A,C), C is the hardest negative in another camera. 
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self, margin=None):
        self.margin=margin
        if margin is not None:
            self.ranking_loss1=nn.MarginRankingLoss(margin=margin[0],reduction="mean")
            self.ranking_loss2=nn.MarginRankingLoss(margin=margin[1],reduction="mean")
        else:
            self.ranking_loss=nn.SoftMarginLoss(reduction="mean")

    def __call__(self,feat,labels,cameras,model=None,paths=None,epoch=0,normalize_feature=False):
        if normalize_feature: # default: don't normalize , distance [0,1]
            feat = normalize(feat,axis=-1)
        dist_mat = euclidean_dist(feat, feat)
        d1, d2, d3, _ = distance_mining(dist_mat, labels, cameras)

        y = d1.new().resize_as_(d1).fill_(1)
        if self.margin is not None:
            l1 = self.ranking_loss1(d2, d1, y)
            l2 = self.ranking_loss2(d3, d2, y)
        else:
            l1 = self.ranking_loss(d2-d1, y)
            l2 = self.ranking_loss(d3-d2, y)
        loss = l2 + l1
        return loss



def distance_mining_global(dist_mat, labels1, labels2, cams1, cams2):

    assert len(dist_mat.size()) == 2
    num_query = dist_mat.size(0)
    num_gallery = dist_mat.size(1)
    is_pos = labels2.expand(num_query, num_gallery).eq(labels1.expand(num_gallery, num_query).t())
    # is_neg=labels.expand(N,N).ne(labels.expand(N,N).t()) # | cameras.expand(N,N).ne(cameras.expand(N,N).t())

    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]  # each anchor's hardest positive
    d1 = d1.squeeze(1)
    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2 = d1.new().resize_as_(d1).fill_(0)
    d3 = d1.new().resize_as_(d1).fill_(0)

    for i in range(num_query):
        _, sorted_inds = torch.sort(dist_mat[i])  # distance ascending
        cam_id = cams1[i]
        B, C = False, False
        for ind in sorted_inds:
            if labels2[ind] == labels1[i]:
                continue
            if (B == False) and (cam_id == cams2[ind]):  # intra-camera hardest negative
                d3[i] = dist_mat[i, ind]
                B = True
            if (C == False) and (cam_id != cams2[ind]):  # inter-camera hardest negative
                d2[i] = dist_mat[i, ind]
                C = True
            if B and C:
                break
    return d1, d2, d3


def distance_mining_global_multi_neg(dist_mat, labels1, labels2, cams1, cams2, negK=5, intra_negK=1):

    assert len(dist_mat.size()) == 2
    num_query = dist_mat.size(0)
    num_gallery = dist_mat.size(1)
    is_pos = labels2.expand(num_query, num_gallery).eq(labels1.expand(num_gallery, num_query).t())

    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]  # each anchor's hardest positive
    d1 = d1.squeeze(1)

    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2 = torch.zeros((num_query, negK)).cuda()   # cross-camera hard negatives

    if intra_negK==1:
        d3 = d1.new().resize_as_(d1).fill_(0)  # intra-camera hard negative
    else:
        d3 = torch.zeros((num_query, intra_negK)).cuda()

    val_inds = []
    for i in range(num_query):
        cc = cams1[i]
        lbl = labels1[i]

        #print('cams1 shape= {}, cams2 shape= {}, labels shape= {}, labels2 shape= {}'.format(cams1.shape, cams2.shape, labels1.shape, labels2.shape))        
        intra_neg_inds = torch.nonzero((cams2==cc) & (labels2!=lbl)).squeeze(-1)

        inter_neg_inds = torch.nonzero(cams2!=cc).squeeze(-1)
        
        if len(intra_neg_inds)==0 or len(inter_neg_inds)==0: continue

        neg1 = torch.topk(dist_mat[i, intra_neg_inds], 1, largest=False)[1]
        d3[i] = dist_mat[i, intra_neg_inds[neg1]]

        neg2 = torch.topk(dist_mat[i, inter_neg_inds], min(negK, len(inter_neg_inds)), largest=False)[1]
        d2[i,0:len(neg2)] = dist_mat[i, inter_neg_inds[neg2]]
        
        val_inds.append(i)

    d1 = d1[val_inds]
    d2 = d2[val_inds]
    d3 = d3[val_inds]
    
    return d1, d2, d3




class MCNL_Loss_global(object):
    """Extends the Multi-camera negative loss to allow comparing two features
       d1=(A,A'), A' is the hardest true positive.
       d2=(A,C), C is the hardest negative in another camera.
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self, margin=None, negK=1, intra_negK=1):
        self.margin = margin
        self.negK = negK
        self.intra_negK = intra_negK
        if margin is not None:
            self.ranking_loss1=nn.MarginRankingLoss(margin=margin[0],reduction="mean")
            self.ranking_loss2=nn.MarginRankingLoss(margin=margin[1],reduction="mean")
        else:
            self.ranking_loss=nn.SoftMarginLoss(reduction="mean")


    def __call__(self, feat1, feat2, labels1, labels2, cams1, cams2, normalize_feature=False):

        if normalize_feature:  # default: don't normalize , distance [0,1]
            feat1 = normalize(feat1, axis=-1)
            feat2 = normalize(feat2, axis=-1)

        distmat = euclidean_dist(feat1, feat2)
        #print('batch labels= ', labels1.cpu().detach())
        #print('batch cams= ', cams1.cpu().detach())
        
        if self.negK == 1:
            d1, d2, d3 = distance_mining_global(distmat, labels1, labels2, cams1, cams2)
            y = d1.new().resize_as_(d1).fill_(1)
            if self.margin is not None:
                l1 = self.ranking_loss1(d2, d1, y)
                l2 = self.ranking_loss2(d3, d2, y)
            else:
                l1 = self.ranking_loss(d2 - d1, y)
                l2 = self.ranking_loss(d3 - d2, y)
            loss = l2 + l1
        else:
            d1, d2, d3 = distance_mining_global_multi_neg(distmat, labels1, labels2, cams1, cams2, self.negK, self.intra_negK)

            if len(d1) == 0: return 0

            y = d1.new().resize_as_(d1).fill_(1)

            l1, l2 = 0, 0
            for i in range(d2.size(1)):  # top-i cross-negative
                d2_one = d2[:, i]
                l1 += self.ranking_loss1(d2_one, d1, y)  # loss between positive and inter-cam negatives    
                if self.intra_negK>1:
                    for j in torch.arange(self.intra_negK):
                        l2 += self.ranking_loss2(d3[:, j], d2_one, y)
                else:
                    l2 += self.ranking_loss2(d3, d2_one, y)            
            loss = l2 + l1
            
        return loss



