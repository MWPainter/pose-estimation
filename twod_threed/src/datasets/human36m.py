#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset


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


class Human36M(Dataset):
    def __init__(self, actions, data_path, use_hg=True, is_train=True):
        """
        :param actions: list of actions to use
        :param data_path: path to dataset
        :param use_hg: use stacked hourglass detections
        :param is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = data_path

        self.is_train = is_train
        self.use_hg = use_hg

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'

        if self.is_train:
            # load train data
            self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file))
            for k2d in self.train_2d.keys():
                k3d = get_3d_key_from_2d_key(k2d)
                num_f, _ = self.train_2d[k2d].shape
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.train_inp.append(self.train_2d[k2d][i])
                    self.train_out.append(self.train_3d[k3d][i])

        else:
            # load test data
            self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                k3d = get_3d_key_from_2d_key(k2d)
                if act not in self.actions:
                    continue
                num_f, _ = self.test_2d[k2d].shape
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[k2d][i])
                    self.test_out.append(self.test_3d[k3d][i])

    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()

        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)
