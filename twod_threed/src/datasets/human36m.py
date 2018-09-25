#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.human36m_dataset import Human36mDataset


TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]



def get_3d_key_from_2d_key(k2d):
    """
    Given k2d of the form (actor number, action, filename)
    :param k2d: The key for the 2d dataset
    :return: k3d, the key for the 3d dataset (the corresponding 3d pose)
    """
    (sub, act, fname) = k2d
    k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k2d
    return k3d



class Human36M(Human36mDataset):
    def __init__(self, actions, data_path, cams_per_frame=4, is_train=True, orthogonal_data_augmentation=False,
                 z_rotations_only=False, dataset_normalization=False, num_joints=32, num_joints_pred=16, flip_prob=0.5,
                 drop_joint_prob=0.0):
        super(Human36M, self).__init__(dataset_path=data_path, cams_per_frame=cams_per_frame, is_train=is_train,
                orthogonal_data_augmentation=orthogonal_data_augmentation, z_rotations_only=z_rotations_only,
                dataset_normalization=dataset_normalization, num_joints=num_joints, num_joints_pred=num_joints_pred,
                flip_prob=flip_prob, drop_joint_prob=drop_joint_prob)

    def get_stat_2d(self):
        return {'mean': self.pose_2d_mean, 'std': self.pose_2d_std, 'dim_use': self.pose_2d_indx_to_use}

    def get_stat_3d(self):
        return {'mean': self.pose_3d_mean, 'std': self.pose_3d_std, 'dim_use': self.pose_3d_indx_to_use}

    def __getitem__(self, index):
        _, _, pose_projected, pose_camera_coords, meta = super(Human36M, self).__getitem__(index)
        return pose_projected, pose_camera_coords, meta
