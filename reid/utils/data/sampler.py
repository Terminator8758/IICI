from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

# for RandomBiCameraSampler:
import copy
import itertools 
from typing import Optional
from . import comm


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4, class_position=1):
        self.data_source = data_source
        #self.class_position = class_posotion
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        #for index, (_, pid, _) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[class_position]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        
        #for index, (_, pid, cam) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[1]
            cam = each_input[2]
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            #_, i_pid, i_cam = self.data_source[i]
            i_pid = self.data_source[i][1]
            i_cam = self.data_source[i][2]
            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:  # as a priority: select images in the same cluster/class, from different cameras (my add)

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:  # otherwise select images in the same camera, or do not select more if it's an outlier (my add)
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)


class ClassUniformlySampler(Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''
    def __init__(self, samples, class_position, k=4, has_outlier=False, cam_num=0):

        self.samples = samples
        self.class_position = class_position
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        id_dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]   # from which index to obtain the label
            if class_index not in list(id_dict.keys()):
                id_dict[class_index] = [index]
            else:
                id_dict[class_index].append(index)
        return id_dict

    def _generate_list(self, id_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []

        dict_copy = id_dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        outlier_cnt = 0
        for key in keys:
            value = dict_copy[key]
            if len(value)==1:  #self.has_outlier and len(value)==1:  # len(value)<=self.cam_num:
                continue
                #sample_list.append(value[0])
                #sample_list.append(value[0])  # repeat the single instance twice (for CSBN)
                #outlier_cnt += 1
            elif len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
        if outlier_cnt > 0:
            print('in Sampler: outlier number= {}'.format(outlier_cnt))
        return sample_list




class ClassUniformlySamplerWithCamStyle(Sampler):
    def __init__(self, samples, class_position, k=4, ori_sample_ratio=0.5):

        self.samples = samples
        self.class_position = class_position
        self.num_instances = k
        self.ori_sample_n = int(k*ori_sample_ratio)
        self.tran_sample_n = k-self.ori_sample_n
        self.class_dict, self.class_dict_ori, self.class_dict_transfer = self._tuple2dict(self.samples)
        self.pids = list(self.class_dict.keys())
        self.num_ids = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_ids).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            value = self.class_dict_ori[pid]
            if len(value)==1:
                continue
            elif len(value) >= self.num_instances:
                value = np.random.choice(value, size=self.ori_sample_n, replace=False)
            else:
                value = np.random.choice(value, size=self.ori_sample_n, replace=True)
            ret.extend(value)
            # sample cam-style transferred image
            value_tran = self.class_dict_transfer[pid]
            value_tran = np.random.choice(value_tran, size=self.tran_sample_n, replace=False)
            ret.extend(value_tran)
        return iter(ret)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return:
        id_dict, {class_index_i: [samples_index1, samples_index2, ...]}
        id_dict_ori_cam, {class_index_i: [samples_index1, samples_index2, ...]}
        id_dict_transfer_cam, {class_index_i: [samples_index5, samples_index6, ...]}
        '''
        id_dict = defaultdict(list)
        id_dict_ori = defaultdict(list)
        id_dict_transfer = defaultdict(list)
        for index, each_input in enumerate(inputs):
            pid = each_input[self.class_position]  # (img_path, pid, camid, target_cam)
            id_dict[pid].append(index)
            transfer_cam = each_input[3]
            if transfer_cam==-1:
                id_dict_ori[pid].append(index)
            else:
                id_dict_transfer[pid].append(index)
        return id_dict, id_dict_ori, id_dict_transfer




class ClassCamAwareSampler(Sampler):
    '''
    random sample according to class label, also make sure at least two classes from one camera are sampled.
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''
    def __init__(self, samples, class_position, k=4, has_outlier=False, cam_num=0, batch_sz=64):

        self.samples = samples
        self.class_position = class_position
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.batch_sz = batch_sz
        self.class_dict, self.cam_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict, self.cam_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        id_dict = {}
        cam_dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]   # from which index to obtain the label
            cam = each_input[2]
            if class_index not in list(id_dict.keys()):
                id_dict[class_index] = [index]
            else:
                id_dict[class_index].append(index)
            # put id into its camera dict
            if cam not in cam_dict:
                cam_dict[cam] = [class_index]
            elif class_index not in cam_dict[cam]:
                cam_dict[cam].append(class_index)

        # for cam_dict, remove ids which have only one image
        for cam in cam_dict.keys():
            cam_dict[cam] = [id for id in cam_dict[cam] if len(id_dict[id])>1]

        # re-check
        for cam in cam_dict.keys():
            for ind, the_id in enumerate(cam_dict[cam]):
                if len(id_dict[the_id]) == 1:
                    print('in _tuple2dict(): there is still singleton class!!')

        return id_dict, cam_dict

    def _generate_list(self, id_dict, cam_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []
        outlier_cnt = 0
        dict_copy = id_dict.copy()

        # random shuffle ids under each camera, then obtain same-cam id pairs, then shuffle them
        same_cam_pairs = []
        cam_dict_copy = cam_dict.copy()
        for cam in cam_dict_copy.keys():
            random.shuffle(cam_dict_copy[cam])
            if len(cam_dict_copy[cam]) % 2 == 0:
                end_pos = len(cam_dict_copy[cam])
            else:
                end_pos = len(cam_dict_copy[cam])-1  # ignore the last id if cannot make a pair
            for start_pos in range(0, end_pos, 2):
                same_cam_pairs.append(cam_dict_copy[cam][start_pos:start_pos+2])  # two same-cam ids at a time

        random.shuffle(same_cam_pairs)

        #new_same_cam_pairs = []
        #for i in range(0, len(same_cam_pairs)-self.batch_sz+1, self.batch_sz):
        #    temp_pairs = np.concatenate(same_cam_pairs[i:i+self.batch_sz])
        #    #random.shuffle(temp_pairs)
        #    new_same_cam_pairs.append(temp_pairs)
        #new_same_cam_pairs = np.concatenate(new_same_cam_pairs)

        sample_ids = np.concatenate(same_cam_pairs)

        # random sample images of each id specified in same_cam_pairs
        for pid in sample_ids:
            value = dict_copy[pid]
            if len(value) == 1:  # self.has_outlier and len(value)==1:  # len(value)<=self.cam_num:
                print('there is still singleton class sampled!')
                continue
            elif len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
        return sample_list


class ClassAndCameraBalancedSampler(Sampler):
    def __init__(self, data_source, num_instances=4, class_position=1):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        # for index, (_, pid, cam) in enumerate(data_source):
        for index, each_input in enumerate(data_source):
            pid = each_input[class_position]
            cam = each_input[2]
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for ii in indices:
            curr_id = self.pids[ii]
            indexes = np.array(self.pid_index[curr_id])
            cams = np.array(self.pid_cam[curr_id])
            uniq_cams = np.unique(cams)
            if len(uniq_cams) >= self.num_instances:  # more cameras than per-class-instances
                sel_cams = np.random.choice(uniq_cams, size=self.num_instances, replace=False)
                for cc in sel_cams:
                    ind = np.where(cams==cc)[0]
                    sel_idx = np.random.choice(indexes[ind], size=1, replace=False)
                    ret.append(sel_idx[0])
            else:
                sel_cams = np.random.choice(uniq_cams, size=self.num_instances, replace=True)
                for cc in np.unique(sel_cams):
                    sample_num = len(np.where(sel_cams == cc)[0])
                    ind = np.where(cams == cc)[0]
                    if len(ind) >= sample_num:
                        sel_idx = np.random.choice(indexes[ind], size=sample_num, replace=False)
                    else:
                        sel_idx = np.random.choice(indexes[ind], size=sample_num, replace=True)
                    for idx in sel_idx:
                        ret.append(idx)
        return iter(ret)



class RandomBiCameraSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.num_instances = num_instances

        self.pid_dic = defaultdict(list)
        self.cid_dic = defaultdict(list)
        self.pid_cid_dic = defaultdict(list)
        pid_cam = defaultdict(set)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.batch_size = mini_batch_size * self._world_size

        for index, info in enumerate(data_source):
            pid = info[1]
            cid = info[2]
            self.pid_dic[pid].append(index)
            pid_cam[pid].add(cid)
            if not pid in self.cid_dic[cid]:
                self.cid_dic[cid].append(pid)

        self.pids = list(self.pid_dic.keys())
        self.cids = list(sorted(self.cid_dic.keys()))
        self.num_identities = len(self.pids)
        self.num_cameras = len(self.cids)
        pids_reco = {}
        total_pids = 0
        for cid in self.cids:
            pids_reco[cid] = len(self.cid_dic[cid])
            total_pids += len(self.cid_dic[cid])
        self.prob = []
        for cid in sorted(pids_reco.keys()):
            self.prob.append(pids_reco[cid] / total_pids)

        self.num_pids_per_batch = mini_batch_size // self.num_instances
        self.num_pids_per_batch_per_camera = self.num_pids_per_batch // 2  #self.num_cameras

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}
            batch_indices = []
            while len(avl_pids) >= self.num_pids_per_batch:
                selected_cids = np.random.choice(self.cids, 2, False, p=self.prob)  # according to id number in each cam
                selected_cids = np.sort(selected_cids)
                for c in selected_cids:  # two cameras
                    batch_indices_temp = []
                    t = self.cid_dic[c]
                    replace = False if len(t) >= self.num_instances else True
                    selected_pids = np.random.choice(t,size=self.num_pids_per_batch_per_camera,replace=replace)  # for each camera: sample 64/8=8 IDs?
                    for pid in selected_pids:
                        # Register pid in batch_idxs_dict if not
                        if pid not in batch_idxs_dict:
                            idxs = copy.deepcopy(self.pid_dic[pid])
                            np.random.shuffle(idxs)
                            batch_idxs_dict[pid] = idxs
                        avl_idxs = batch_idxs_dict[pid]
                        for num,i in enumerate(avl_idxs):
                            batch_indices_temp.append(i)
                    replace = False if len(batch_indices_temp) >= int(self.batch_size/2) else True
                    batch_indices_temp = np.random.choice(batch_indices_temp, int(self.batch_size/2), replace)
                    # batch size=128, 2 cameras, 64 images per-camera (if random sampling, how could positive pairs be generated?)
                    batch_indices.extend(batch_indices_temp)

                yield from reorder_index(batch_indices, self._world_size)

                batch_indices = []


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def reorder_index(batch_indices, world_size):
    r"""Reorder indices of samples to align with DataParallel training.
    In this order, each process will contain all images for one ID, triplet loss
    can be computed within each process, and BatchNorm will get a stable result.
    Args:
        batch_indices: A batched indices generated by sampler
        world_size: number of process
    Returns:

    """
    mini_batchsize = len(batch_indices) // world_size
    reorder_indices = []
    for i in range(0, mini_batchsize):
        for j in range(0, world_size):
            reorder_indices.append(batch_indices[i + j * mini_batchsize])
    return reorder_indices


