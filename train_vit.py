from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from reid import datasets
from reid.models.resnet_fast_reid import resnet_nl_with_transformer
from reid.models.resnet_fast_reid import resnet
from reid.models.sct_memory import SCT_Memory
from reid import models
from reid import sct_trainers
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import ClassUniformlySampler
from reid.utils.data.preprocessor import Preprocessor, TrainPreprocessor
from reid.utils.logging import Logger
from reid.utils.osutils import str2bool
from sklearn.cluster import KMeans
import scipy.io as sio



def get_intra_camera_split(class_centers, class_camera, min_class_thresh=50):
    sub_camera = np.zeros(class_camera.shape, class_camera.dtype)
    uniq_cams = np.unique(class_camera)
    count = 0
    for c in uniq_cams:
        percam_inds = np.where(class_camera == c)[0]
        num_subcams = int(round(len(percam_inds)*1.0/min_class_thresh))
        if num_subcams==1 and len(percam_inds)>min_class_thresh:
            num_subcams = 2

        if num_subcams > 1:  # to divide
            percam_class_feat = class_centers[percam_inds]
            km = KMeans(n_clusters=num_subcams, random_state=10).fit(percam_class_feat)
            temp_label = km.labels_
            sub_camera[percam_inds] = temp_label + count
            count += (temp_label.max()+1)
        else:
            sub_camera[percam_inds] = count
            count += 1
    return sub_camera



def get_data(name, data_dir, SCT=False, overlap_ratio=0):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root, SCT=SCT, overlap_ratio=overlap_ratio)
    return dataset



def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, 
                     trainset=None, colorjitter=False, dataset_name=None, self_norm=True):

    if self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    if colorjitter:
        brightness = 0.2
        contrast = 0.15 if args.dataset_name == 'MSMT17' else 0

        aug_transform = T.Compose([
                 T.Resize((height, width), interpolation=3),
                 T.RandomHorizontalFlip(p=0.5),
                 T.Pad(10),
                 T.RandomCrop((height, width)),
                 T.ColorJitter(brightness=brightness, contrast=contrast, saturation=0, hue=0), 
                 T.ToTensor(),
                 normalizer,
                 T.RandomErasing(probability=0.6, mean=[0.485, 0.456, 0.406]) 
             ])
    else:
        aug_transform = None

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    assert(num_instances > 0)

    sampler = ClassUniformlySampler(train_set, class_position=1, k=num_instances)

    train_loader = IterLoader(DataLoader(
        TrainPreprocessor(train_set, root=dataset.images_dir, transform=train_transformer, 
            aug_transform=aug_transform), batch_size=batch_size, num_workers=workers, 
            sampler=sampler, shuffle=False, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(dataset, height, width, batch_size, workers, testset=None, self_norm=True):

    if self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


     
def create_model(args):
    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=0.3,
                          pretrained_path='pretrained/vit_small_ics_cfs_lup.pth', hw_ratio=2, conv_stem=True)

    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main(args):
    #args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'train_vit.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset_name, args.data_dir, SCT=args.SCT, overlap_ratio=args.overlap_ratio)  # note here
    dataset.train = sorted(dataset.train)

    # get propagate loader and test loader
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, self_norm=args.self_norm)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, testset=dataset.train, self_norm=args.self_norm)
        
    # Create model
    model = create_model(args)

    # Create memory
    memory = SCT_Memory(temp=args.temp, momentum=args.momentum, bg_knn=args.bg_knn, 
                        has_intra_cam_loss=args.has_intra_cam_loss, has_mcnl_loss=args.has_mcnl_loss, 
                        mcnl_negK=args.mcnl_negK, dataset_name=args.dataset_name, arch=args.arch).cuda()

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]


    if args.has_detr:
        detr_model = resnet_nl_with_transformer.Detr_transformer(num_patch=12, tran_hidden_dim=256)
        detr_model.cuda()
        detr_model = nn.DataParallel(detr_model)
        detr_params = [{"params": [value]} for _, value in detr_model.named_parameters() if value.requires_grad]
        params += detr_params
    else:
        detr_model = None


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # initialize class memory
    print('==> Extract feature for all images')    
    features, _ = extract_features(model, cluster_loader, print_freq=200)    
    features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in dataset.train], 0)

    all_img_cams = torch.tensor([c for _, _, c, _ in dataset.train])
    gt_labels = np.array([lbl for _, lbl, _, _ in dataset.train])
    num_ids = len(set(gt_labels))

    class_centers = torch.zeros(num_ids, features.size(1))
    class_camera = torch.zeros(num_ids)
    gt_labels = torch.from_numpy(gt_labels)
    for ii in range(num_ids):
        idx = torch.nonzero(gt_labels == ii).squeeze(-1)
        class_centers[ii] = features[idx].mean(0)
        class_camera[ii] = all_img_cams[idx[0]]
        num_cams = len(torch.unique(all_img_cams[idx]))
        assert(num_cams==1)
    class_centers = F.normalize(class_centers.detach(), dim=1).cuda()
    print('  initializing class memory feature with shape {}...'.format(class_centers.shape))
    memory.class_memory = class_centers.detach()
    if args.dataset_name == 'MSMT17':
        memory.class_original_camera = class_camera.long().cuda()
    else:
        memory.class_camera = class_camera.long().cuda()


    train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                    args.num_instances, iters, trainset=dataset.train, 
                                    colorjitter=args.has_aug_transform, dataset_name=args.dataset_name, self_norm=args.self_norm)
    train_loader.new_epoch()


    # Trainer
    trainer = sct_trainers.SCT_Trainer(model, memory, arch=args.arch, has_aug_transform=args.has_aug_transform, detr_model=detr_model)


    for epoch in range(args.epochs):

        temp_class_subcam = ''
        if args.split_subcamera:
            
            print('Loading pretrained model generated sub-cameras:')
            class_subcam = sio.loadmat('msmt_pretrained/MSMT_id_subcam_label_epoch_'+str(epoch)+'.mat')['data'][0]
            #print('  class_subcam shape= {}, dtype={}'.format(class_subcam.shape, class_subcam.dtype))
            temp_class_subcam = torch.from_numpy(class_subcam).cuda()
            memory.class_camera = temp_class_subcam

            all_subcam_ids = []
            for cc in np.unique(class_subcam):
                subcam_cls = np.where(class_subcam == cc)[0]
                all_subcam_ids.append(len(subcam_cls))

            print('  ID number under each sub-camera= ', all_subcam_ids)
            print('  generating sub-camera label with {} sub-cameras from {} to {}'
                .format(len(np.unique(class_subcam)), class_subcam.min(), class_subcam.max()))
             

        # train an epoch
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader), class_subcam=temp_class_subcam)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            print('==> Epoch {} test: '.format(epoch))
            evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

        lr_scheduler.step()

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Camera Training re-ID")
    # data
    parser.add_argument('--dataset_name', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_instances', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--SCT', type=str2bool, default=True, help="Single Camera Training setting")
    parser.add_argument('--overlap_ratio', type=float, default=0, help="ID overlap ratio")
    # model
    parser.add_argument('--arch', type=str, default='vit_small')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pool_type', type=str, default='gempool')
    parser.add_argument('--has_detr', type=str2bool, default=False)
    parser.add_argument('--self_norm', type=bool, default=True)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step_size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # loss
    parser.add_argument('--has_intra_cam_loss', type=str2bool, default=True)
    parser.add_argument('--has_mcnl_loss', type=str2bool, default=True)
    parser.add_argument('--mcnl_negK', type=int, default=20)
    parser.add_argument('--has_aug_transform', type=str2bool, default=True)
    parser.add_argument('--split_subcamera', type=str2bool, default=True)
    parser.add_argument('--bg_knn', type=int, default=50)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update rate for the memory bank")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    args = parser.parse_args()

    main(args)



