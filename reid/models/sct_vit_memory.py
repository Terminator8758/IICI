from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np

from .cross_entropy_loss import CrossEntropyLabelSmooth
from .mcnl_loss import NegativeLoss
from .original_mcnl_loss import MCNL_Loss, MCNL_Loss_global, MCNL_Loss_softmax
from .mcnl_cluster_loss import MCNL_cluster_loss
from .triplet_loss_stb import TripletLoss
from .sup_contrastive_loss import SupConLoss_clear

torch.autograd.set_detect_anomaly(True)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_crosscam_data(x, y, cam, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    percam_other_inds = {}
    cam = cam.cpu()
    for c in torch.unique(cam):
        percam_other_inds[int(c)] = torch.nonzero(cam!=c).squeeze(-1)

    index = torch.zeros(len(x), dtype=torch.int64)
    for i in range(len(x)):
        index[i] = np.random.choice(percam_other_inds[int(cam[i])], 1)[0]
    index = index.detach().cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class KDLoss(nn.Module):
    def __init__(self, t):
        super(KDLoss, self).__init__()
        self.t = t
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.t, dim=1)
        target = F.softmax(target/self.t, dim=1)
        loss = self.kl_div(log_p, target)*(self.t**2)/input.size(0)
        return loss


class ExemplarMemory(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        #print('memory update: labels= ', indexes.cpu().detach())
        # momentum update
        for x, y in zip(inputs, indexes):
            #if ctx.momentum[i] > 0:
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
            
        return grad_inputs, None, None, None


class SCT_Memory(nn.Module):
    def __init__(self, temp=0.07, momentum=0.2, all_img_cams='', has_cluster_loss=False, per_camera_loss=False,
                 bg_knn=50, posK=3, cluster_hard_mining=False, has_intra_cam_loss=False, has_intra_cam_mining_loss=False,
                 has_cross_cam_mining_loss=False, label_smoothing=False, num_classes=0, has_mcnl_loss=False, mcnl_negK=1, choose_cam_wise_negK=False,
                 has_triplet_loss=False, has_crosscam_soft_loss=False, feature_mixup=False, compute_per_cam_center=False,
                 cluster_loss_with_multi_pos=False, has_augmented_memory_loss=False, finegrain_class=False, has_instance_memory_loss=False,
                 has_batch_contrast_loss=False, has_local_memory=False, has_intra_inv_loss=False,  has_global_inv_loss=False, has_intra_cam_merge_loss=False,
                 has_decoupled_loss=False, has_mix_half_loss=False, has_gray_prototype_loss=False, local_cluster_contrast=False):
        super(SCT_Memory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum
        self.unique_cameras = np.unique(all_img_cams)
        self.all_pseudo_label = ''

        self.has_cluster_loss = has_cluster_loss
        self.cluster_hard_mining = cluster_hard_mining
        self.per_camera_loss = per_camera_loss
        self.has_intra_cam_loss = has_intra_cam_loss
        self.has_cross_cam_mining_loss = has_cross_cam_mining_loss
        self.has_intra_cam_mining_loss = has_intra_cam_mining_loss
        self.has_mcnl_loss = has_mcnl_loss

        self.bg_knn = bg_knn
        self.posK = posK

        self.class_memory = ''
        self.class_camera = ''
        self.proxy_gt_label = ''

        self.label_smoothing = label_smoothing
        if self.label_smoothing:
            self.smooth_ce_criterion = CrossEntropyLabelSmooth(num_classes=num_classes)

        if self.has_mcnl_loss:
            self.MCNL_criterion = MCNL_Loss_global(margin=(0.1, 0.1), negK=mcnl_negK, intra_negK=1, decreasing_margin=False)
            #self.MCNL_criterion = MCNL_cluster_loss(margin=(0.1, 0.1), negK=mcnl_negK, intra_negK=1)
            self.MCNL_criterion_sub = MCNL_Loss_global(margin=(0.1, 0.1), negK=10, intra_negK=1, decreasing_margin=False)
            self.MCNL_criterion_batch = MCNL_Loss_global(margin=(0.1, 0.1), negK=5, intra_negK=1)  

        self.has_triplet_loss = has_triplet_loss
        if self.has_triplet_loss:
            self.trip_criterion = TripletLoss(margin=0.3, metric='euclidean')

        self.has_crosscam_soft_loss = has_crosscam_soft_loss

        self.kd_criterion = KDLoss(t=1)  # changed the temperature

        self.feature_mixup = feature_mixup

        self.compute_per_cam_center = compute_per_cam_center
        if self.compute_per_cam_center:
            self.per_cam_centers = {}

        self.cluster_loss_with_multi_pos = cluster_loss_with_multi_pos

        self.has_augmented_memory_loss = has_augmented_memory_loss
        if self.has_augmented_memory_loss:
            self.augmented_memory = ''  # per-camera memory for each class, eg: [memory_of_cam0, memory_of_cam1, ...]
            self.mse_criterion = nn.MSELoss().cuda()

        self.finegrain_class = finegrain_class

        self.has_instance_memory_loss = has_instance_memory_loss
        if has_instance_memory_loss:
            self.instance_memory = ''
            self.instance_camera = ''
            self.instance_labels = ''

        self.has_batch_contrast_loss = has_batch_contrast_loss
        if has_batch_contrast_loss:
            self.SupConLoss_criterion = SupConLoss_clear(0.07)

        self.has_local_memory = has_local_memory
        self.has_intra_inv_loss = has_intra_inv_loss
        self.has_global_inv_loss = has_global_inv_loss
        self.has_intra_cam_merge_loss = has_intra_cam_merge_loss
        self.has_decoupled_loss = has_decoupled_loss
        self.has_mix_half_loss = has_mix_half_loss

        self.has_gray_prototype_loss = has_gray_prototype_loss
        self.local_cluster_contrast = local_cluster_contrast

        self.SupConLoss_criterion = SupConLoss_clear(0.07)


    def forward(self, features, labels, cams, epoch, batch_ind=-1, gap_feat='', augment_features='', tgt_labels='', mixstyle_feat='', local_feat='', perm_index=''):
        
        loss = torch.tensor(0.).to(torch.device('cuda'))
        #print('batch labels= ', labels.cpu().detach())
        #print('batch cams= ', cams.cpu().detach())        
        #print('batch tgt_labels= ', tgt_labels.cpu().detach())
        #print('batch feat shape= {}, memory feat shape= {}'.format(features.shape, self.class_memory.shape))
        loss_change_epoch = 100
        #temp_memory = self.class_memory.detach().clone()
            
        if self.has_mcnl_loss:
            temp_memory = self.class_memory.detach().clone()
            memory_id_label = torch.arange(len(self.class_memory)).cuda()
            #print('batch mcnl feat shape= {}, label shape= {}, cam shape= {}'.format(mcnl_feat.shape, mcnl_label.shape, mcnl_cam.shape))
            #loss += self.MCNL_criterion_batch(mcnl_feat, mcnl_feat, mcnl_label, mcnl_label, mcnl_cam, mcnl_cam)
            #loss += self.MCNL_criterion(mcnl_feat, self.class_memory.detach().clone(), mcnl_label, torch.arange(len(self.class_memory)).cuda(), mcnl_cam, self.class_camera)
            #loss += self.MCNL_criterion_batch(features, features, labels, labels, cams, cams)
            loss += self.MCNL_criterion_batch(features, features, labels, labels, tgt_labels, tgt_labels)  # for original cameras
            #loss += self.MCNL_criterion(features, temp_memory, labels, memory_id_label, cams, self.class_camera)    
            loss += self.MCNL_criterion(features, temp_memory, labels, memory_id_label, tgt_labels, self.class_original_camera) # for original camera
            #loss += self.MCNL_criterion(features, temp_memory, labels, memory_id_label, cams, self.class_camera, id_cluster_label=self.id_cluster_label)
            
            for c in torch.unique(tgt_labels):  # compute under each original camera
                inds = torch.nonzero(tgt_labels==c).squeeze(-1)
                memo_inds = torch.nonzero(self.class_original_camera==c).squeeze(-1)
                if len(torch.unique(self.class_camera[memo_inds])) > 1:
                    weight = 1.0*len(inds)/len(cams)
                    loss += weight * self.MCNL_criterion_sub(features[inds], temp_memory[memo_inds], labels[inds], memory_id_label[memo_inds], cams[inds], self.class_camera[memo_inds])
            

            if len(local_feat) > 0:  # transformer feature
                print('using transformer feature...')
                transformer_loss = 0
                if len(local_feat) < 5:
                    #print('batch local feat len= {}, element0 shape= {}'.format(len(local_feat), local_feat[0].shape))
                    for j in range(len(local_feat)):
                        local_feat_p = F.normalize(local_feat[j])
                        transformer_loss += self.MCNL_criterion_batch(local_feat_p, local_feat_p, labels, labels, cams, cams)
                else:
                    #transformer_loss += self.MCNL_criterion_batch(local_feat, local_feat, labels, labels, tgt_labels, tgt_labels)
                    transformer_loss += self.MCNL_criterion_batch(local_feat, local_feat, labels, labels, cams, cams)
                loss += transformer_loss
        #return loss
        
        #if self.has_batch_contrast_loss:
        #    loss += self.SupConLoss_criterion(features, labels)  # supervised contrastive loss considering camera?
       
        if self.has_mix_half_loss and len(perm_index)>0:
            perm_labels = labels[perm_index]  # label of the mixed half image
            perm_cams = cams[perm_index]  # camera of the mixed half image
            self.per_camera_classes = {int(j): torch.nonzero(self.class_camera == j).squeeze(-1) for j in torch.unique(self.class_camera)}
            #print('perm_labels= ', perm_labels.cpu().detach())
            cluster_ori_score = torch.zeros((len(features), len(self.class_memory))).cuda()
            unmix_ind = torch.nonzero(labels == perm_labels).squeeze(-1)
            if len(unmix_ind) > 0:  # ori img without mix-half
                score1 = ExemplarMemory.apply(features[unmix_ind], labels[unmix_ind], self.class_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
                cluster_ori_score[unmix_ind] = score1
            mix_ind = torch.nonzero(labels != perm_labels).squeeze(-1)
            if len(mix_ind) > 0:  # img mix-halfed with another img
                score2 = torch.matmul(features[mix_ind], self.class_memory.t().detach().clone())
                cluster_ori_score[mix_ind] = score2
            cluster_score = cluster_ori_score / self.temp
            #print('batch unmix num= {}, mix num= {}'.format(len(unmix_ind), len(mix_ind)))

        elif self.has_gray_prototype_loss:
            gray_ind = torch.arange(1, len(features), 2).cuda()
            gray_features = features[gray_ind]
            rgb_ind = torch.arange(0, len(features), 2).cuda()
            features = features[rgb_ind]
            cluster_ori_score = ExemplarMemory.apply(features, labels, self.class_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
            gray_score = ExemplarMemory.apply(gray_features, labels, self.gray_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
            cluster_score = cluster_ori_score / self.temp
            gray_score = gray_score / self.temp
        else:
            cluster_ori_score = ExemplarMemory.apply(features, labels, self.class_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
            cluster_score = cluster_ori_score / self.temp
        
        if self.local_cluster_contrast:
            img_index = tgt_labels
            local_labels = self.local_clusters[img_index]  # [labels]  
            local_score = ExemplarMemory.apply(features, local_labels, self.cluster_proxies, torch.tensor(self.momentum).to(torch.device('cuda'))) 
            local_score = local_score / self.temp
            #print('batch local label= ', local_labels.cpu().detach())
            #print('batch local_label= ', local_labels.cpu().detach())
            local_loss = 0.0001*F.cross_entropy(local_score, local_labels)  # just to update the proxy feat
            local_loss += ((features - self.cluster_proxies[local_labels].detach())**2).mean(0).sum()
            #local_con_loss = 0
            #for j in range(len(cluster_score)):
            #    local_clu = self.local_clusters[labels[j]]
            #    local_cls_inds = torch.nonzero(self.local_clusters == local_clu).squeeze(-1)
            #    if len(local_cls_inds) > 1:
            #        local_target = torch.zeros(len(local_cls_inds)).cuda()
            #        pos = torch.nonzero(local_cls_inds==labels[j]).squeeze(-1)
            #        local_target[pos] = 1
            #        local_con_loss += -1.0 * (F.log_softmax(cluster_score[j, local_cls_inds].unsqueeze(0), dim=1) * local_target.unsqueeze(0)).sum()
            #loss += local_con_loss/len(cluster_score)
            #
            #clu_proxies = torch.zeros(features.shape, dtype=features.dtype).cuda()
            #for kk in torch.unique(local_labels):
            #    img_inds = torch.nonzero(local_labels==kk).squeeze(-1)
            #    cls_inds = torch.nonzero(self.local_clusters == kk).squeeze(-1)
            #    clu_proxies[img_inds] = self.class_memory[cls_inds].mean(0).detach()
            #clu_proxies = F.normalize(clu_proxies)
            #local_loss = ((features - clu_proxies)**2).mean(0).sum()
            loss += local_loss

        #if self.has_local_memory and (len(local_feat) > 0):  # transformer feature
        #    local_score = ExemplarMemory.apply(local_feat, labels, self.local_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
        #    local_score = local_score / self.temp
        
        all_intra_loss = []
        for c in torch.unique(cams):

            if epoch >= loss_change_epoch: continue

            inds = torch.nonzero(cams==c).squeeze(-1)
            percam_score = cluster_score[inds]
            percam_label = labels[inds]

            # cluster contrastive loss
            if self.cluster_hard_mining:
                loss += self.get_cluster_mining_loss(percam_score, percam_label, c)
                #loss += self.get_cluster_mining_loss_weighted(percam_score, percam_label, epoch)
            elif self.has_cluster_loss:
                if self.label_smoothing:
                    loss += self.smooth_ce_criterion(percam_score, percam_label)
                else:
                    loss += F.cross_entropy(percam_score, percam_label)
            elif self.cluster_loss_with_multi_pos and epoch>=5:
                loss += self.get_cluster_loss_with_multi_pos(percam_score, features[inds], percam_label, c)

            # per-camera supervised contrastive loss
            if self.has_intra_cam_loss:
                if self.finegrain_class:
                    loss += self.get_intra_loss_fine_grained(percam_score, percam_label, c)
                else:
                    weight = 0.05  # len(inds)*1.0/len(cams)
                    loss += weight*self.get_intra_loss_all_cams(percam_score, percam_label, c)
                    #loss += weight*self.get_pseudo_intra_loss_all_cams(percam_score, percam_label, tgt_labels[inds[0]])  

                    if self.has_local_memory and (len(local_feat) > 0):
                        percam_local_score = local_score[inds]
                        loss += self.get_intra_loss_all_cams(percam_local_score, percam_label, c)
                    
            if self.has_intra_cam_mining_loss:
                loss += self.get_intra_loss_hard_mining(percam_score, percam_label, c)

            if self.has_intra_cam_merge_loss and epoch>=0:
                loss += self.get_intra_loss_in_merged_cam(percam_score, percam_label, c)

            if self.has_intra_inv_loss:
                loss += self.get_intra_invariant_loss(percam_score, percam_label)
            
            if self.has_gray_prototype_loss:
                loss += 0.5*self.get_gray_prototype_loss(gray_score[inds], percam_label, c)

            if self.has_mix_half_loss:
                mix_loss1, mix_loss2 = self.get_mix_half_loss(percam_score, percam_label, perm_labels[inds], perm_cams[inds], c)
                loss += mix_loss1  # per-camera ori-img loss
                loss += mix_loss2  #*(len(inds)*2/len(cams))  # per-batch mix-img loss

            if self.has_cross_cam_mining_loss:
                loss += self.get_cross_camera_consistency_loss(features[inds], percam_score, c)
                #loss += self.get_cross_camera_mining_loss(percam_score, percam_label, c, 0)

            if self.has_crosscam_soft_loss:
                loss += (1.0*len(inds)/len(features)) * self.get_crosscam_soft_label_loss(cluster_ori_score[inds], percam_label, c)  # use original score

            if self.has_decoupled_loss:
                loss += self.get_decoupled_loss(percam_score, percam_label, c)

            if self.has_global_inv_loss and epoch>=10:
                #temp_score = torch.matmul(gap_feat[inds], temp_memory.t()) / self.temp
                #print('checking: temp_score[1,0:20]= {}, percam_score[1,0:20]= {}'.format(temp_score[1,0:20].cpu().detach(), percam_score[1,0:20].cpu().detach()))
                loss += self.get_global_invariant_loss(percam_score, percam_label)

            if self.has_triplet_loss:
                percam_gap_feat = gap_feat[inds]
                loss += self.trip_criterion(percam_gap_feat, percam_gap_feat, percam_gap_feat, percam_label, percam_label, percam_label)

            #if self.has_mcnl_loss:
                #loss += self.MCNL_criterion(features[inds], self.class_memory.detach(), percam_label, torch.arange(len(self.class_memory)).cuda(), cams[inds], self.class_camera)
                #if len(inds) > 4:
                #    mcnl_loss = self.MCNL_criterion_batch(local_feat[inds], local_feat, percam_label, labels, cams[inds], cams)
                #    print('batch per-camera mcnl loss= ', mcnl_loss.item())
                #    loss += mcnl_loss
                    #loss += self.MCNL_criterion_batch(gap_feat[inds], gap_feat, percam_label, labels, cams[inds], cams)

                #print('batch per-cam tgt_labels= ', tgt_labels[inds].cpu().detach())
                #print('self.class_original_camera: ')
                #loss += self.MCNL_criterion(features[inds], self.class_memory.detach(), percam_label, torch.arange(len(self.class_memory)).cuda(), tgt_labels[inds], self.class_original_camera)
                #val_inds = torch.nonzero(tgt_labels==tgt_labels[inds[0]]).squeeze(-1)
                #if len(val_inds) > 4:  # len(inds) > 4:
                #    if len(gap_feat) > 0:
                #        loss += self.MCNL_criterion_batch(gap_feat[inds], gap_feat, percam_label, labels, tgt_labels[inds], tgt_labels)  # cams[inds], cams)

            if self.has_instance_memory_loss:
                loss += self.MCNL_criterion(features[inds], temp_inst_memory, percam_label, self.instance_labels, cams[inds], self.instance_camera)
                # update instance memory
                uniq_lbl = torch.unique(percam_label)
                for lbl in uniq_lbl:
                    img_inds = torch.nonzero(labels == lbl).squeeze(-1)
                    rand_two = torch.randperm(len(img_inds))[0:2]
                    this_lbl = int(lbl)
                    self.instance_memory[this_lbl*2: this_lbl*2+2] = features[img_inds[rand_two]].detach().clone()

            if self.has_augmented_memory_loss:
                augment_loss = 0
                uniq_lbl = torch.unique(percam_label)
                for lbl in uniq_lbl:
                    img_inds = torch.nonzero(labels == lbl).squeeze(-1)        
                    aug_feat = [self.augmented_memory[kk][lbl] for kk in self.augmented_memory.keys() if kk!=c]
                    aug_feat = torch.vstack(aug_feat)
                    for jj in img_inds:
                        augment_loss += ((features[jj] - aug_feat)**2).sum()
                augment_loss /= (len(percam_label)*(len(self.augmented_memory.keys())-1))
                loss += augment_loss
                
                ## version 0: global multi-positive hard-mining contrastive loss
                percam_aug_scores = []
                for kk in self.augmented_memory.keys():
                    if kk!=c:
                        cs_score = torch.matmul(features[inds], self.augmented_memory[kk].t()) / self.temp
                        percam_aug_scores.append(cs_score)
                percam_aug_scores = torch.hstack(percam_aug_scores)
                # percam_aug_scores = torch.cat(percam_aug_scores, dim=1)
                loss += self.get_all_cam_augmented_loss(percam_score, percam_label, percam_aug_scores)
                ## version 1: positive feature consistency loss
                augment_loss = 0
                uniq_lbl = torch.unique(percam_label)
                for lbl in uniq_lbl:
                    img_inds = torch.nonzero(labels == lbl).squeeze(-1)
                    for cid in self.unique_cameras:
                        aug_feat = augment_features[cid][img_inds]
                        augment_loss += (1-torch.matmul(features[img_inds], aug_feat.t())).mean()
                loss += augment_loss/len(uniq_lbl)
                ## version 2: global contrastive loss taking augmented feature as positive proxies
                augmented_proxies = {}
                for lbl in torch.unique(percam_label):
                    augmented_proxies[int(lbl)] = []
                    img_inds = torch.nonzero(labels == lbl).squeeze(-1)
                    lbl2 = int(lbl)
                    for cid in self.unique_cameras:
                        if cid != int(c):
                            aug_feat = augment_features[cid][img_inds].mean(dim=0)
                            assert(aug_feat.norm()>0)
                            aug_feat /= aug_feat.norm()
                            augmented_proxies[lbl2].append(aug_feat)
                    augmented_proxies[lbl2] = torch.vstack(augmented_proxies[lbl2]).t()

                augment_loss = 0.5*self.get_global_augmented_loss_v3(percam_score, percam_label, features[inds], augmented_proxies)
                loss += augment_loss
                                
        if self.feature_mixup:
            loss += self.get_cluster_loss_with_mixup(features, labels, cams)

        return loss
        


    def get_intra_loss_all_cams(self, inputs, labels, cam):
        
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        #print('cam {}: intra-subcam class number= {}'.format(int(cam), len(intra_classes)))
        #intra_gt_label = self.class_gt_label[intra_classes]
        for i in range(len(inputs)):
            #pos = torch.nonzero(intra_gt_label==self.class_gt_label[labels[i]]).squeeze(-1)
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            targets[i, pos] = 1.0 / len(pos)

        log_probs = F.log_softmax(inputs[:, intra_classes], dim=1)
        if self.label_smoothing:
            targets = (1-0.1)*targets + 0.1/len(intra_classes)
        loss = (-targets*log_probs).mean(0).sum()
        #loss = -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()
         
        return loss



    def get_intra_loss_in_merged_cam(self, inputs, labels, cam):
        loss = 0
        intra_classes = self.ids_in_merged_cams[int(cam)]
        temp_score = inputs[:, intra_classes].detach().clone()
        #targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        for i in range(len(inputs)):
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            temp_score[i, pos] = 1000
            _, top_inds = torch.topk(temp_score[i], min(len(intra_classes), 50))
            sel_score = inputs[i, intra_classes[top_inds]]
            #print('sel_score shape= ', sel_score.shape)
            sel_target = torch.zeros((len(sel_score),), dtype=sel_score.dtype).cuda()
            #print('sel_score shape= {}, sel_target shape= {}'.format(sel_score.shape, sel_target.shape))
            sel_target[0] = 1.0
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
            #targets[i, pos] = 1.0
        #loss += -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()
        loss /= len(inputs)
        return loss


    def get_pseudo_intra_loss_all_cams(self, inputs, labels, ori_cam):
        loss = 0
        intra_classes = torch.nonzero(self.class_original_camera == ori_cam).squeeze(-1)
        #print('cam {}: intra_classes number= {}'.format(ori_cam, len(intra_classes)))
        targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        for i in range(len(inputs)):
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            #print('positive index= ', pos.cpu().detach())
            targets[i, pos] = 1.0
        loss += -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()

        return loss


    def get_pseudo_intra_loss_all_cams_v2(self, inputs, labels, ori_cam):
        loss = 0
        # compute intra-camera inter-subcam loss  [0,2,4],[1,3,5],[6,8,10],[7,9]
        # self.class_pseudo_camera: sub-camera of all ID classes;
        intra_classes = torch.nonzero(self.class_original_camera == ori_cam).squeeze(-1)
        intra_sub_classes = self.class_pseudo_camera[intra_classes]
        uniq_subcams = torch.unique(intra_sub_classes)
        subcam_id_dic = {}
        for cc in uniq_subcams:
            subcam_id_dic[int(cc)] = torch.nonzero(self.class_pseudo_camera==cc).squeeze(-1)
            #print('subcam_id_dic: sub-cam {} has {} IDs'.format(cc, len(subcam_id_dic[int(cc)])))
        for i in range(len(inputs)):
            for cc in subcam_id_dic.keys():
                neg_ids = subcam_id_dic[cc]
                neg_ids = neg_ids[neg_ids!=labels[i]]
                sel_input = torch.cat((inputs[i, neg_ids], inputs[i,labels[i]].unsqueeze(dim=0)))
                target = torch.zeros((len(sel_input),), dtype=sel_input.dtype).cuda()
                target[-1] = 1.0
                loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()

        loss /= len(inputs)
        return loss


    def get_intra_loss_all_cams_with_transfer(self, inputs, labels, cc, transferred_cam):
        loss = 0
        intra_cls = torch.nonzero(self.class_camera == cc).squeeze(-1)
        memory_cls = torch.arange(len(self.class_memory)).cuda()
        #print('transfer cams= ', transferred_cam.cpu().detach())
        for i in range(len(inputs)):
            if transferred_cam[i] == -1:  # an original image
                sel_score = inputs[i, intra_cls]
                pos_ind = torch.nonzero(intra_cls == labels[i]).squeeze(-1)
                target = torch.zeros(len(intra_cls)).cuda()
                target[pos_ind] = 1.0
            else:
                pos_ind = torch.nonzero(memory_cls==labels[i]).squeeze(-1)
                neg_ind = torch.nonzero(self.class_camera == transferred_cam[i]).squeeze(-1)
                sel_score = torch.cat((inputs[i, pos_ind], inputs[i, neg_ind]))
                target = torch.zeros(1+len(neg_ind)).cuda()
                target[0] = 1.0
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    # each ID is split into fine-grained sub-classes
    def get_intra_loss_fine_grained(self, inputs, labels, cam):
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        intra_gt_label = self.proxy_gt_label[intra_classes]
        targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        loss = 0
        for i in range(len(inputs)):
            target_lbl = self.proxy_gt_label[labels[i]]
            pos = torch.nonzero(intra_gt_label==target_lbl).squeeze(-1)
            targets[i, pos] = 1.0/len(pos)
            #print('sub-id label= {}, positive proxies= {}'.format(labels[i].cpu().detach(), pos.cpu().detach()))
        loss += -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()
        return loss


    # compute invariance loss of two subsets within the same camera
    def get_intra_invariant_loss(self, inputs, labels):
        loss = 0
        uniq_classes = torch.unique(labels)
        for cls in uniq_classes:
            inds = torch.nonzero(labels==cls).squeeze(-1)
            intra_envs = self.intra_env_set[int(cls)]  # each class has 2 envs, similar ones and dissimilar ones
            temp_loss = []
            if len(intra_envs) == 1: continue  # added here

            for j in range(len(intra_envs)):
                neg_inds = intra_envs[j]
                #print('neg_inds shape= ', neg_inds.shape)
                sel_input = torch.cat((inputs[inds][:, cls].unsqueeze(1), inputs[inds][:, neg_inds]), dim=1)
                target = torch.zeros(sel_input.shape).cuda()
                target[:, 0] = 1
                loss_env = -1.0 * (F.log_softmax(sel_input, dim=1) * target).mean(0).sum()
                temp_loss.append(loss_env)
            loss += torch.var(torch.stack(temp_loss))

        loss = loss/len(uniq_classes)
        #print('per-camera intra inv loss= ', loss.item())
        return loss


    def get_intra_loss_hard_mining(self, inputs, labels, cam):
        loss = 0
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        temp_score = inputs[:, intra_classes].detach().clone()
        intr_cls_num = len(intra_classes)
        intra_gt_label = self.class_gt_label[intra_classes]
        for i in range(len(inputs)):
            pos = torch.nonzero(intra_gt_label==self.class_gt_label[labels[i]]).squeeze(-1)
            #pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)  # original version
            temp_score[i, pos] = 1000
            _, top_inds = torch.topk(temp_score[i], min(intr_cls_num, self.bg_knn+len(pos)))
            sel_score = inputs[i, intra_classes[top_inds]]
            sel_target = torch.zeros((len(sel_score)), dtype=sel_score.dtype).to(torch.device('cuda'))
            sel_target[0:len(pos)] = 1.0/len(pos) 
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    def get_intra_loss_all_cams_smooth(self, inputs, labels, cam, tgt_labels):
        # tgt labels: -1 for original images, otherwise is the transfer camera label (to which style)
        loss = 0
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        targets = torch.zeros((len(inputs), len(intra_classes))).cuda()
        for i in range(len(inputs)):
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            if tgt_labels[i] == -1:
                targets[i, pos] = 1.0
            else:
                targets[i] = 0.1/len(intra_classes)  # label smoothing
                targets[i, pos] += 0.9 
        loss += -1.0 * (F.log_softmax(inputs[:, intra_classes], dim=1) * targets).mean(0).sum()
        return loss


    def get_gray_prototype_loss(self, scores, labels, cam):
        # score: similarity between grayscale image features and gray memory feature
        #score = features.mm(self.gray_prototypes.t())
        intra_classes = torch.nonzero(self.class_camera == cam).squeeze(-1)
        targets = torch.zeros((len(scores), len(intra_classes))).cuda()
        for i in range(len(scores)):
            #pos = torch.nonzero(intra_gt_label==self.class_gt_label[labels[i]]).squeeze(-1)
            pos = torch.nonzero(intra_classes==labels[i]).squeeze(-1)
            targets[i, pos] = 1.0 / len(pos)
        loss = -1 * (F.log_softmax(scores[:, intra_classes], dim=1) * targets).mean(0).sum()
        '''
        temp_score = scores.detach().clone()
        for i in range(len(scores)):
            temp_score[i, labels[i]] = 10000

        _, top_inds = torch.topk(temp_score, self.bg_knn+1, dim=1)
        sel_score = scores.gather(1, top_inds)
        # ori_pos_score = (features * self.class_memory[labels]).sum(dim=1)
        #sel_score = torch.cat((sel_score, ori_pos_score.unsqueeze(-1)), dim=1)
        print('image-to-gray-prototype score= ', sel_score.cpu().detach())
        print('top gray prototypes camera source= ')
        for i in range(len(top_inds)):
            print(self.class_camera[top_inds[i]].cpu())
        target = torch.zeros(sel_score.shape, dtype=sel_score.dtype).cuda()
        target[:, 0] = 1
        loss = -1 * (F.log_softmax(sel_score, dim=1) * target).mean(0).sum()
        #print('gray feature loss= ', loss.item())
        '''
        return loss


    def get_mix_half_loss(self, score, labels, perm_labels, perm_cams, cam):
        N = len(score)
        intra_classes = self.per_camera_classes[int(cam)]  #torch.nonzero(self.class_camera == cam).squeeze(-1)
        loss1, loss2 = 0, 0
        count1, count2 = 0, 0
        single_pos_temp, all_pos_temp = [], []
        for i in range(N):
            if perm_labels[i]==labels[i]:  # original image without mixing, or mixing of the same id
                targets = torch.zeros((len(intra_classes),), dtype=score.dtype).cuda()
                pos = torch.nonzero(intra_classes == labels[i]).squeeze(-1)
                targets[pos] = 1.0
                count1 += 1
                #print('ori-img: pos similarity score= ', score[i, intra_classes[pos]].cpu().detach())
                loss1 += -1 * (F.log_softmax(score[i, intra_classes].unsqueeze(0), dim=1) * targets.unsqueeze(0)).sum()

            elif perm_labels[i] in intra_classes:  # mixhalf with intra-camera img
                targets = torch.zeros((len(intra_classes),), dtype=score.dtype).cuda()
                pos = torch.nonzero(intra_classes == labels[i]).squeeze(-1)
                mix_pos = torch.nonzero(intra_classes == perm_labels[i]).squeeze(-1)
                targets[pos] = 0.5
                targets[mix_pos] = 0.5
                count2 += 1
                loss2 += -1 * (F.log_softmax(score[i, intra_classes].unsqueeze(0), dim=1) * targets.unsqueeze(0)).sum()

            else:  # mixhalf with cross-camera img
                sel_classes = torch.cat((intra_classes, self.per_camera_classes[int(perm_cams[i])]))
                sel_score = score[i, sel_classes]   #* self.temp  # here recover ori similarity!
                targets = torch.zeros((len(sel_score),), dtype=sel_score.dtype).cuda()
                pos = torch.nonzero((sel_classes==labels[i]) | (sel_classes==perm_labels[i])).squeeze(-1)        
                targets[pos] = 0.5
                count2 += 1
                #all_pos_temp.append(pos)
                #print('mixup img: pos similarity score= ', sel_score[pos].cpu().detach())    
                loss2 += -1 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * targets.unsqueeze(0)).sum()

        loss1 /= count1
        loss2 /= count2
        #loss = loss1 + 0.5*loss2
        #print('cam= {}: count1= {}, count2= {}'.format(cam, count1, count2))
        #print('image num of unmix, intra-mix and inter-mix= ', temp_count)
        return loss1, loss2



    def get_all_cam_augmented_loss_v2(self, percam_score, percam_label, percam_aug_scores):
        # percam_aug_scores: positive score between query and its camstyle-augmented features, shape=(C,batch_sz)
        pos_num = len(percam_aug_scores)
        percam_score = torch.cat((percam_score, percam_aug_scores.t()), dim=1)
        soft_label = torch.zeros(percam_score.shape, dtype=percam_score.dtype).cuda()
        for i in range(len(soft_label)):
            soft_label[i, percam_label[i]] = 1.0/(1+pos_num)
            soft_label[i, -pos_num:] = 1.0/(1+pos_num)
            #print('self-pos sim= {}, style-augment sim= {}'.format(percam_score[i, percam_label[i]].cpu().detach(), percam_score[i, -pos_num:].cpu().detach()))
        loss = -1.0 * (F.log_softmax(percam_score, dim=1) * soft_label).mean(0).sum()
        return loss


    def get_global_augmented_loss_v3(self, percam_score, percam_label, percam_feat, augmented_proxies):
        loss = 0
        for i in range(len(percam_score)):
            lbl = int(percam_label[i])
            aug_pos_num = augmented_proxies[lbl].size(1)
            augmented_score = torch.matmul(percam_feat[i], augmented_proxies[lbl]) / self.temp
            #print('aug_pos_num= {}, augmented_score= {}'.format(aug_pos_num, augmented_score.cpu().detach()))
            augmented_score = torch.cat((percam_score[i], augmented_score))

            soft_label = torch.zeros(len(augmented_score), dtype=percam_score.dtype).cuda()
            soft_label[lbl] = 1.0 / (1 + aug_pos_num)
            soft_label[-aug_pos_num:] = 1.0 / (1 + aug_pos_num)
            loss += -1.0 * (F.log_softmax(augmented_score.unsqueeze(0), dim=1) * soft_label.unsqueeze(0)).sum()
        loss /= len(percam_score)
        return loss


    def get_all_cam_augmented_loss(self, percam_score, percam_label, percam_aug_scores):
        
        percam_aug_scores = torch.cat((percam_score, percam_aug_scores), dim=1)
        # soft_label = torch.zeros(percam_aug_scores.shape, dtype=percam_aug_scores.dtype).cuda()
        temp_score = percam_aug_scores.detach().clone()

        loss = 0
        for k in range(len(percam_label)):
            ori_pos = percam_label[k]
            pos_index = [ori_pos + m * len(self.class_memory) for m in range(0, len(self.augmented_memory))]
            #print('self.unique_cameras= {}, pos_index= {}'.format(self.unique_cameras, pos_index))
            pos_index = torch.tensor(pos_index)
            temp_score[k, ori_pos] = 2000
            temp_score[k, pos_index[1:]] = 1000
            _, top_inds = torch.topk(temp_score[k], self.bg_knn+len(pos_index))
            sel_score = percam_aug_scores[k, top_inds]
            sel_target = torch.zeros((len(sel_score)), dtype=sel_score.dtype).to(torch.device('cuda'))
            sel_target[0] = 0.5
            sel_target[1:len(pos_index)] = 0.5/(len(pos_index)-1)
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(percam_score)
        #print('per-camera augmented_loss= ', augmented_loss.item())
        return loss


    def get_cluster_mining_loss(self, cluster_score, labels, cam=-1):
        temp_score = cluster_score.detach().clone()
        loss = 0
        for i in range(len(cluster_score)):
            temp_score[i, labels[i]] = -1000  # mark the positives
            _, top_inds = torch.topk(temp_score[i], self.bg_knn)
            sel_score = torch.cat((cluster_score[i, top_inds], cluster_score[i, labels[i]].unsqueeze(0)))
            intra_neg = torch.nonzero(self.class_camera[top_inds]==cam).squeeze(-1)  # should exclude positive class
            cross_neg = torch.nonzero(self.class_camera[top_inds]!=cam).squeeze(-1)
            # weighted label where cross-cam neg has larger weight
            sel_target = torch.zeros((len(sel_score)), dtype=sel_score.dtype).to(torch.device('cuda'))
            sel_target[-1] = 0.9  # positive class
            w1 = 0.1/(len(cross_neg)+0.5*len(intra_neg))
            sel_target[intra_neg] = 0.5*w1
            sel_target[cross_neg] = w1
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(cluster_score)
        return loss


    # compute invariance loss of intra-camera and inter-camera environmet
    def get_global_invariant_loss(self, inputs, labels):
        loss, contrast_loss = 0, 0
        uniq_classes = torch.unique(labels)
        for cls in uniq_classes:
            inds = torch.nonzero(labels == cls).squeeze(-1)
            global_envs = self.global_env_set[int(cls)]  # each class has 2 envs, intra-camera negs and inter-camera negs
            temp_loss = []
            for j in range(len(global_envs)):
                neg_inds = global_envs[j]
                #assert(cls not in neg_inds)
                #print('per-camera hard negatives len= ', len(neg_inds))
                sel_input = torch.cat((inputs[inds][:, cls].unsqueeze(1), inputs[inds][:, neg_inds]), dim=1)
                target = torch.zeros(sel_input.shape).cuda()
                target[:, 0] = 1
                loss_env = -1.0 * (F.log_softmax(sel_input, dim=1) * target).mean(0).sum()
                temp_loss.append(loss_env)
                #if j==1:  # inter-camera contrastive loss
                #    contrast_loss += loss_env
            loss += torch.var(torch.stack(temp_loss))

        loss = loss/len(uniq_classes)
        #loss += contrast_loss/len(uniq_classes)
        #print('global invariant loss= ', loss.item())
        return loss


    def get_cluster_mining_loss_weighted(self, cluster_score, labels, epoch):

        temp_score = cluster_score.detach().clone()

        mask_topk = int((100-epoch)*1.0)
        for i in range(len(cluster_score)):
            temp_score[i, labels[i]] = -1000  # mark the positives
            _, mask_inds = torch.topk(temp_score[i], mask_topk)
            temp_score[i, mask_inds] = -1000  # mask top negatives
        #temp_score = torch.matmul(self.class_memory[labels], self.class_memory.t()).detach().clone()

        percam_hard_neg = []
        uniq_cams = torch.unique(self.class_camera)
        for j in uniq_cams:
            percam_neg = torch.nonzero(self.class_camera == j).squeeze(-1)
            _, hard_neg_ind = torch.topk(temp_score[:, percam_neg], 10, dim=1)  # 10 neg in each cam
            percam_hard_neg.append(percam_neg[hard_neg_ind])
            #hard_neg_ind = torch.argmax(temp_score[:, percam_neg], dim=1)
            #percam_hard_neg.append(percam_neg[hard_neg_ind].unsqueeze(-1))
        percam_hard_neg = torch.cat(percam_hard_neg, dim=1)  # batchSize x numCams, per-camera hardest negative

        #all_cross_neg = torch.nonzero(self.class_original_camera != cam).squeeze(-1)
        #all_intra_neg = torch.nonzero(self.class_original_camera == cam).squeeze(-1)
        loss = 0
        for i in range(len(cluster_score)):
            #temp_score[i, labels[i]] = -1000  # mark the positives
            #_, top_inds = torch.topk(temp_score[i], self.bg_knn)
            #sel_score = torch.cat((cluster_score[i, top_inds], cluster_score[i, labels[i]].unsqueeze(-1)))
            #_, intra_inds = torch.topk(temp_score[i, all_intra_neg], sel_negK)
            #_, cross_inds = torch.topk(temp_score[i, all_cross_neg], sel_negK)  # per-camera negative?

            sel_score = torch.cat((cluster_score[i, percam_hard_neg[i]], cluster_score[i, labels[i]].unsqueeze(-1)))
            #inter_score = cluster_score[i, all_cross_neg[cross_inds]]
            #target = cluster_score[i, all_intra_neg[intra_inds]]
            #inter_score = torch.cat((cluster_score[i, all_cross_neg[cross_inds]], cluster_score[i, labels[i]].unsqueeze(-1)))
            #target = torch.cat((cluster_score[i, all_intra_neg[intra_inds]], cluster_score[i, labels[i]].unsqueeze(-1)))
            #target = target.detach()

            #loss += self.kd_criterion(inter_score.unsqueeze(0), target.unsqueeze(0))
            #loss += torch.pow(inter_score-target, 2).mean(0).sum()
            #weights = F.softmax(1-temp_score[i, all_cross_neg[cross_inds]], dim=0)
            #sel_score = torch.cat((cluster_score[i, all_cross_neg[cross_inds]], cluster_score[i, labels[i]].unsqueeze(-1)))
            sel_target = torch.zeros((len(sel_score)), dtype=sel_score.dtype).to(torch.device('cuda'))
            sel_target[-1] = 1
            #sel_target[0:len(cross_inds)] = 0.05/len(cross_inds)
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(cluster_score)
        return loss


    def get_cluster_loss_with_multi_pos(self, cluster_score, features, labels, cam):
        other_centers = []
        uniq_cams = torch.unique(self.class_camera)
        for cc in uniq_cams:
            if cc!=cam:
                other_centers.append(self.per_cam_centers[int(cc)])
        other_centers = torch.vstack(other_centers).detach().cuda()

        ori_pos_memo = self.class_memory[labels].detach().clone()
        loss = 0
        for i in range(len(cluster_score)):
            # generate cross-camera positive memory feature
            gene_pos_memo = 0.8*ori_pos_memo[i] + 0.2*other_centers
            gene_pos_memo = F.normalize(gene_pos_memo, dim=1)  # normalize the mixed centers
            gene_pos_score = torch.matmul(features[i], gene_pos_memo.t())
            gene_pos_score /= self.temp
            sel_score = torch.cat((cluster_score[i], gene_pos_score))
            sel_target = torch.zeros((len(sel_score)), dtype=sel_score.dtype).to(torch.device('cuda'))
            sel_target[labels[i]] = 1.0/len(uniq_cams)
            sel_target[-len(uniq_cams)+1:] = 1.0/len(uniq_cams)
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(cluster_score)
        return loss


    def get_cluster_loss_with_mixup(self, features, labels, cams):
        temp_memory = self.class_memory.detach().clone()
        mixed_feat, label1, label2, lam = mixup_data(features, labels)
        #mixed_feat, label1, label2, lam = mixup_crosscam_data(features, labels, cams, alpha=1.0)
        mixed_score = torch.mm(mixed_feat, temp_memory.t())  # batch_sz * num_classes
        mixed_score /= self.temp
        mixed_label = torch.zeros(mixed_score.shape, dtype=mixed_score.dtype).cuda()
        for i in range(len(features)):
            mixed_label[i, label1[i]] = lam
            mixed_label[i, label2[i]] = 1-lam
        loss = -1.0 * (F.log_softmax(mixed_score, dim=1) * mixed_label).mean(0).sum()
        return loss


    # cross-camera contrastive loss with label smoothing
    def get_cross_camera_mining_loss(self, cluster_score, labels, cam=-1, epsilon=0):
        temp_score = cluster_score.detach().clone()
        loss = 0
        cross_classes = torch.nonzero(self.class_camera != cam).squeeze(-1)
        sel_target = torch.zeros(self.bg_knn+1, dtype=cluster_score.dtype).to(torch.device('cuda'))
        sel_target[0] = 1.0 - epsilon  # smoothed label
        sel_target[1:] = epsilon/self.bg_knn

        for i in range(len(cluster_score)):
            temp_score[i, labels[i]] = 1000  # mark the positives
            sel_classes = torch.cat((labels[i].unsqueeze(0), cross_classes))
            _, top_inds = torch.topk(temp_score[i, sel_classes], self.bg_knn+1)
            sel_score = cluster_score[i, top_inds]
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()

        loss /= len(cluster_score)
        return loss


    # consistency loss between feature and its mixed fake-positives
    # features: bottleneck features without L2 normalization
    def get_cross_camera_consistency_loss(self, features, scores, cam):
        #uniq_cams = torch.unique(self.class_camera)
        #temp_score = scores.detach().clone()
        inter_classes = torch.nonzero(self.class_camera != cam).squeeze(-1)
        all_mix_feat = torch.zeros(features.shape, dtype=features.dtype).cuda()
        
        for i in range(len(features)):
            _, top_cls = torch.topk(scores[i, inter_classes], 20)  # all camera or each camera?
            mix_feat = 0.5*features[i] + 0.5*self.class_memory[top_cls].mean(dim=0)
            all_mix_feat[i] = mix_feat.detach()
        all_mix_feat = F.normalize(all_mix_feat)
        # mse loss between (features, all_mix_feat):
        loss = ((features - all_mix_feat.detach())**2).mean(0).sum()
        #print('crosscam consistency loss= ', loss.item())
        return loss
        

    # cross-camera contrastive loss with online similarity as soft target
    def get_crosscam_soft_label_loss(self, cluster_score, labels, cam, cam_wise=False):
        #temp_lbl = torch.arange(len(self.class_memory)).cuda()
        #temp_score = cluster_score.detach().clone()
        #for i in range(len(temp_score)):
        #    temp_score[i, labels[i]] = -10000
        mask = torch.zeros(cluster_score.shape, dtype=cluster_score.dtype).detach().cuda()
        for i in range(len(cluster_score)):
            mask[i, labels[i]] = -10000
        cluster_score += mask

        intra_cls = torch.nonzero(self.class_camera == cam).squeeze(-1) 
        inter_cls = torch.nonzero(self.class_camera != cam).squeeze(-1)
        #soft_targets = torch.matmul(self.class_memory[labels], self.class_memory[intra_cls].t())
        soft_targets = torch.sort(cluster_score[:, intra_cls], dim=1)[0]  # similarity in ascending order
        soft_targets = soft_targets[:, 1:]
        #topk = min(len(intra_cls)-1, 50)
        #soft_targets = soft_targets[:, -topk:]  # exclude positive score
        #soft_targets += 0.1
        #soft_targets[soft_targets>1] = 1

        # all other cameras or per-camera manner:
        if cam_wise:    
            all_inter_cls = []  #torch.nonzero(self.class_camera != cam).squeeze(-1)
            for cc in torch.unique(self.class_camera):
                if cc!=cam:
                    ind = torch.nonzero(self.class_camera == cc).squeeze(-1)
                    all_inter_cls.append(ind)
        else:
            all_inter_cls = [torch.nonzero(self.class_camera != cam).squeeze(-1)]

        loss = 0
        for inter_cls in all_inter_cls:
            inter_score = torch.sort(cluster_score[:, inter_cls], dim=1)[0]
            num = min(soft_targets.size(1), len(inter_cls))
            inter_score = inter_score[:, -num:]
            #loss += ((inter_score - soft_targets.detach())**2).mean(0).sum() # MSE loss, absolute sim optimization
            #loss += ((inter_score - soft_targets)**2).mean(0).sum()  # relative sim optimization 
            loss += self.kd_criterion(inter_score, soft_targets[:, -num:].detach())
        #loss = -1.0 * (F.log_softmax(cluster_score[:, crosscam_cls], dim=1) * soft_targets).mean(0).sum()
        return loss/len(all_inter_cls)


    def get_decoupled_loss(self, cluster_score, labels, cam):
        crosscam_cls = torch.nonzero(self.class_camera != cam).squeeze(-1)
        #mask = torch.zeros(len(cluster_score), len(crosscam_cls))
        exp_sim = torch.exp(cluster_score[:, crosscam_cls])
        neg_loss = torch.log(exp_sim.sum()) 
        all_pos_score = cluster_score[torch.arange(len(cluster_score)), labels]
        weight = 2 - F.softmax(all_pos_score*self.temp/0.5, dim=0) * len(cluster_score)
        pos_loss = -(all_pos_score*weight).sum()
        loss = (neg_loss+pos_loss)/len(cluster_score)
        #print('decoupled loss= ', loss.item())
        return loss


    def get_multi_sync_pos_loss(self, features, cams):
        uniq_cams = torch.unique(self.class_original_camera)
        eps = 1e-6
        self.percam_mu, self.percam_sigma = [], []
        for cc in uniq_cams:
            inds = torch.nonzero(self.class_original_camera==cc).squeeze()
            percam_memory = self.class_memory[inds].detach().clone()
            mu = percam_memory.mean(dim=0, keepdim=True)
            var = percam_memory.var(dim=0, keepdim=True)
            sig = (var + eps).sqrt()
            self.percam_mu.append(mu)  # keep track of self.percam_mu, self.percam_sigma
            self.percam_sigma.append(sig)

        # generate cross-camera positives
        temp_feat = features.detach().clone()
        batch_mu = temp_feat.mean(dim=0, keepdim=True)
        batch_var = temp_feat.var(dim=0, keepdim=True)
        batch_sig = (batch_var + eps).sqrt()
        temp_feat = (temp_feat - batch_mu) / batch_sig

        loss = 0
        for c in torch.unique(cams):
            percam_inds = torch.nonzero(cams==c).squeeze()
            sync_pos_score = torch.zeros((len(percam_inds), len(uniq_cams) - 1)).cuda()
            #print('sync_pos_score shape= ', sync_pos_score.shape)
            percam_temp_feat = temp_feat[percam_inds]  # per-camera feature
            cnt = 0
            for j in uniq_cams:
                j = int(j)
                if j!=c:
                    sync_feat1 = percam_temp_feat*self.percam_sigma[j] + self.percam_mu[j] # per-camera feat transfer to cam j
                    sync_feat1 = F.normalize(sync_feat1)
                    temp_score = (features[percam_inds]*sync_feat1).sum(dim=1)
                    #print('c= {}, j= {}, temp_score shape= {}'.format(c.cpu(), j, temp_score.shape))
                    sync_pos_score[:, cnt] = 2-2*temp_score  # euclidean dist
                    cnt += 1
            loss += sync_pos_score.sum()
        loss = 0.5*loss/len(features)
        return loss
