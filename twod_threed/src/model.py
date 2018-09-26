#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn

from utils import data_utils


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)





class _LinearBlock(nn.Module):
    """
    A residual block used as part of the "Linear Model", which really is a res net with fully connected layers.
    """
    def __init__(self, linear_size, p_dropout=0.5):
        super(_LinearBlock, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out





class LinearModel(nn.Module):
    """
    A fully connected res net.
    Michael: LinearModel is a very poor name for this... (haven't gotten around to changing it in the code)
    """
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 dataset_normalized_input=False,
                 input_size=32,     # 16 * 2
                 output_size=51):   # 17 * 3
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # If we are normalizing per instance, then the hips input/output is fixed to zero (so we should ignore it)
        # In this case, in self.forward, we chop the first two values off the input, and prepend zeros to the output
        self.advertised_input_size = input_size
        self.advertised_output_size = output_size
        self.instance_normalized_input = not dataset_normalized_input
        if self.instance_normalized_input:
            input_size -= 2
            output_size -= 3

        # 2d joints
        self.input_size =  input_size
        # 3d joints
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(_LinearBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # flatten if input is (batch_size, num_joints, 2) to (batch_size, 2 * num_joints)
        # if there are any errors below, then the input isn't as expected
        batch_size = x.size(0)
        flatten = (x.dim() == 3)
        if flatten:
            x = x.view((batch_size, self.advertised_input_size))

        # If istance normalized, cut off the first coordinate
        if self.instance_normalized_input:
            x = x[:,2:]

        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        # If instance normalized, re-introduce the zeroed hip joint
        # Also help the network out by renormalizing std dev of joint distances to 1, as we know the targets have this
        if self.instance_normalized_input:
            new_y = torch.zeros(batch_size, self.advertised_output_size).cuda()
            new_y[:,3:] = y
            std = data_utils.std_distance_torch_3d(new_y)
            y = new_y / std.view(-1,1)

        # Unflatten
        if flatten:
            y.view((batch_size, -1, 3))

        return y





class Discriminator(nn.Module):
    """
    A discriminator network (with the same architecture other as the "LinearModel"

    Used as part of the CycleGAN, for discriminating in both 2D and 3D poses.
    """
    def __init__(self, dimension=2, num_joints=16, linear_size=1024, num_stage=1, p_dropout=0.5):
        super(Discriminator, self).__init__()

        # params
        self.dim = dimension
        self.num_joints = num_joints
        self.input_size = dimension * num_joints
        self.linear_size = linear_size
        self.num_stage = num_stage
        self.p_dropout = p_dropout

        # Input stem
        self.linear_in = nn.Linear(self.input_size, linear_size)
        self.batch_norm_in = nn.BatchNorm1d(self.linear_size)
        self.relu = nn.ReLU(inplace=True)

        # Output layer
        self.linear_out = nn.Linear(linear_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Hidden stages
        self.dropout = nn.Dropout(self.p_dropout)
        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(_LinearBlock(linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

    def forward(self, x):
        # Input stem (assumes input of shape (batch_size, <other_dims> (prod=inputsize))
        y = x.contiguous().view((-1, self.input_size))
        y= self.linear_in(y)
        y = self.batch_norm_in(y)
        y = self.relu(y)
        y = self.dropout(y)

        # Hidden stages
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Linear + sigmoid out to give answer in range [0,1]
        y = self.linear_out(y)
        return self.sigmoid(y)





class ProjectNet(nn.Module):
    """
    PyTorch nn.Module implementating a trainable weak perspective projection.

    Used for CycleGAN, 3D->2D. (Currently not used).
    """
    def __init__(self, Tx=0, Ty=0, Tz=0, camera_z=-10, focal_length=5, sx=1.0, sy=1.0, tx=0.0, ty=0.0):
        super(ProjectNet, self).__init__()

        self.translate_3d = nn.Parameter(torch.Tensor([Tx, Ty, Tz]))
        self.camera_z = nn.Parameter(torch.Tensor([camera_z]))
        self.focal_length = nn.Parameter(torch.Tensor([focal_length]))
        self.scale_2d = nn.Parameter(torch.Tensor([sx, sy]))
        self.translate_2d = nn.Parameter(torch.Tensor([tx, ty]))



    def forward(self, x):
        """
        Projects a set of 3D points to a 2D plane. Perspecive projection, not orthogonal projection.

        We go through the following steps:
        1. Parameterized 3D translation + translate (everything) so camera origin is 0
        2. Perspective projection onto plane going through (and normal to) (0,0,f)

        :param x: Input of shape (S1, ..., Sk, 3*j) or (S1,...,Sk, j, 3) of 3D points where j = num joints
        :return: Output of projected points, with shape (S1, ..., Sk, 2*j) or (S1, ..., Sk, j, 2) where j = num joints
        """
        y = x.view((-1, 3))
        batch_size = y.size()[0]

        # N.B. Clones are necessary to avoid inplace operations (which make backward pass impossible)
        y += self.translate_3d
        y[:,2] = y[:,2].clone() - self.camera_z
        y *= self.focal_length.repeat(batch_size, 3) / torch.unsqueeze(y[:,2].clone(), 1).repeat(1,3)
        y[:,:2] = y[:,:2].clone() * self.scale_2d.repeat(batch_size,1) + self.translate_2d.repeat(batch_size, 1)

        outshape = list(x.size())
        outshape[-1] = 2 * (outshape[-1] // 3)
        return y[:,:2].contiguous().view(outshape)#.view(x.size())#.contiguous().view(x.size())[:,:2]
