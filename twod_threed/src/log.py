#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import os
import torch


class Logger(object):
    """
    Logger object to log values to a file
    """
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if not title else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        """
        Set the names of the things that we are logging
        """
        if self.resume:
            pass
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, member, mem_type):
        """
        Log the values in the list "member", with types "mem_types". Member[i] is the logged value for self.names[i]
        """
        assert len(self.names) == len(member), '# of data does not match title'
        for index, mem in enumerate(member):
            if mem_type[index] == 'int':
                self.file.write("{}".format(mem))
            else:
                self.file.write("{0:.5f}".format(mem))
            self.file.write('\t')
            self.numbers[self.names[index]].append(mem)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()



def save_ckpt(state, ckpt_path, is_best=True):
    """
    Save a checkpoint (and save the best checkpoint w.r.t. validation loss)
    """
    if is_best:
        file_path = os.path.join(ckpt_path, 'ckpt_best.pth.tar')
        torch.save(state, file_path)
    file_path = os.path.join(ckpt_path, 'ckpt_last.pth.tar')
    torch.save(state, file_path)
