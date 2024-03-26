from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
from torch.cuda import amp
from .original_mcnl_loss import MCNL_Loss_global

torch.autograd.set_detect_anomaly(True)



class ExemplarMemory(autograd.Function):

    @staticmethod
    #@amp.custom_fwd
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    #@amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        
        return grad_inputs, None, None, None



class SCT_Memory(nn.Module):
    def __init__(self, temp=0.07, momentum=0.2, bg_knn=50, has_intra_cam_loss=False, 
                 has_mcnl_loss=False, mcnl_negK=1, dataset_name='MSMT17', arch='resnet50_nl'):
        super(SCT_Memory, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum
        self.bg_knn = bg_knn

        self.has_intra_cam_loss = has_intra_cam_loss
        self.has_mcnl_loss = has_mcnl_loss
        self.dataset_name = dataset_name
        self.arch = arch

        self.class_memory = ''
        self.class_camera = ''
        self.class_original_camera = ''

        if self.has_mcnl_loss or self.has_per_env_mcnl_loss:
            self.MCNL_criterion = MCNL_Loss_global(margin=(0.1, 0.1), negK=mcnl_negK, intra_negK=1)
            self.MCNL_criterion_batch = MCNL_Loss_global(margin=(0.1, 0.1), negK=10, intra_negK=1)
            self.MCNL_criterion_sub = MCNL_Loss_global(margin=(0.1, 0.1), negK=5, intra_negK=1)


    def forward(self, features, labels, cams, epoch, batch_ind=-1, tgt_labels='', aug_features='', trans_feat=''):

        loss = torch.tensor(0.).to(torch.device('cuda'))
        #print('batch labels= ', labels.cpu().detach())
        #print('batch cams= ', cams.cpu().detach())        
        #print('batch tgt_labels= ', tgt_labels.cpu().detach())
        #print('batch features shape= {}, aug_features shape= {}'.format(features.shape, aug_features.shape))

        temp_memory = self.class_memory.detach().clone()

        # ========================== MCNL loss ===========================
        if self.has_mcnl_loss:
            mcnl_loss = 0
            memory_id_label = torch.arange(len(self.class_memory)).cuda()
            mcnl_weight = 0.1 if self.arch=='vit_small' else 1

            if self.dataset_name == 'MSMT17':
                # compute MCNL loss under each original camera
                mcnl_loss += mcnl_weight * self.MCNL_criterion_batch(features, features, labels, labels, tgt_labels, tgt_labels)

                mcnl_loss += mcnl_weight * self.MCNL_criterion(features, temp_memory, labels, memory_id_label, tgt_labels, self.class_original_camera)

                for c in torch.unique(tgt_labels):  
                    inds = torch.nonzero(tgt_labels==c).squeeze(-1)
                    memo_inds = torch.nonzero(self.class_original_camera==c).squeeze(-1)
                    if len(torch.unique(self.class_camera[memo_inds])) > 1:
                        weight = mcnl_weight * 1.0*len(inds)/len(cams)
                        mcnl_loss += weight * self.MCNL_criterion_sub(features[inds], temp_memory[memo_inds], labels[inds],
                                                                 memory_id_label[memo_inds], cams[inds], self.class_camera[memo_inds])
                if len(trans_feat) > 0:
                    loss += 1.0 * self.MCNL_criterion_batch(trans_feat, trans_feat, labels, labels, tgt_labels, tgt_labels)

            else:
                mcnl_loss += mcnl_weight * self.MCNL_criterion_batch(features, features, labels, labels, cams, cams)  # cams is original camera for market1501
                #print('labels len= {}, memory_id_label len= {}, cams len= {}, self.class_camera len= {}'.format(len(labels), len(memory_id_label), len(cams), len(self.class_camera)))
                mcnl_loss += mcnl_weight * self.MCNL_criterion(features, temp_memory, labels, memory_id_label, cams, self.class_camera)
                
                if len(trans_feat) > 0:
                    loss += 0.5 * self.MCNL_criterion_batch(trans_feat, trans_feat, labels, labels, cams, cams)
            
            loss += mcnl_loss
        # =================================================================

        cluster_ori_score = ExemplarMemory.apply(features, labels, self.class_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
        cluster_score = cluster_ori_score / self.temp
        
        if len(aug_features) > 0:
            aug_score = torch.matmul(aug_features, temp_memory.t()) / self.temp

        for c in torch.unique(cams):
            inds = torch.nonzero(cams==c).squeeze(-1)
            percam_score = cluster_score[inds]
            percam_label = labels[inds]
            weight = 1.0*len(inds)/len(cams) if self.arch=='vit_small' else 1

            # per-camera supervised contrastive loss
            if self.has_intra_cam_loss:                           
                loss += weight * self.get_intra_loss_all_cams(percam_score, percam_label, c)

                if len(aug_features) > 0:
                    loss += weight * self.get_intra_loss_all_cams(aug_score[inds], percam_label, c)
                        
        return loss
        


    def get_intra_loss_all_cams(self, inputs, labels, cam):
        loss = 0
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        for i in range(len(inputs)):
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            targets[i, pos] = 1.0
        loss += -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()
        return loss


