from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

from scipy.io import loadmat

import torch
import torch.utils.data as data

from utils.osutils import *
from stacked_hourglass.pose.utils.imutils import *
from stacked_hourglass.pose.utils.transforms import *


class Mpii(data.Dataset):
    """
    Dataset that produces img, 2d pose, meta triples. Where meta contains lots of additional information
    (such as additional persons poses + headbox information and so on)
    """
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, sigma=1, scale_factor=0.25, \
                 rot_factor=30, label_type='Gaussian', mean=None, stddev=None, augment_data=True, args=None):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.augment_data = augment_data
        self.add_random_masking = args is not None and args.add_random_masking

        # Args for when there is random masking
        if self.add_random_masking:
            self.mask_prob = args.mask_prob
            self.orientation_prob = args.orientation_prob
            self.mean_valued_prob = args.mean_valued_prob
            self.max_cover_ratio = args.max_cover_ratio
            self.noise_std = args.noise_std

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        # Store the training and validation indices into the complete dataset
        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)

        # Quite hacky, but when life gives you lemons...
        # Marge joint visibility data and headbox data into self.anno
        # We only have this for the validation set, and we have to trust the authors that headboxes_src is correct
        # Shape of headboxes_src is (2,2,validation_dataset_size)
        dict = loadmat('stacked_hourglass/evaluation/data/detections_our_format.mat')
        if not self.is_train:
            headboxes_src = dict['headboxes_src']
            for i in range(len(self.valid)):
                a = self.anno[self.valid[i]]
                a['headbox'] = headboxes_src[:,:,i]

        # Dictionary of joints (indices of joints we're interested in)
        dataset_joints = dict['dataset_joints']
        self.joint_idxs = {
            'head': np.where(dataset_joints == 'head')[1][0],
            'lsho': np.where(dataset_joints == 'lsho')[1][0],
            'lelb': np.where(dataset_joints == 'lelb')[1][0],
            'lwri': np.where(dataset_joints == 'lwri')[1][0],
            'lhip': np.where(dataset_joints == 'lhip')[1][0],
            'lkne': np.where(dataset_joints == 'lkne')[1][0],
            'lank': np.where(dataset_joints == 'lank')[1][0],
            'rsho': np.where(dataset_joints == 'rsho')[1][0],
            'relb': np.where(dataset_joints == 'relb')[1][0],
            'rwri': np.where(dataset_joints == 'rwri')[1][0],
            'rkne': np.where(dataset_joints == 'rkne')[1][0],
            'rank': np.where(dataset_joints == 'rank')[1][0],
            'rhip': np.where(dataset_joints == 'rhip')[1][0],
        }

        # Avoid work computing new mean + stddev if we can
        if mean is not None and stddev is not None:
            self.mean, self.std = mean, stddev
        else:
            self.mean, self.std = self._compute_mean()


    def _compute_mean(self):
        """
        Helper function to compuete the mean and std of the dataset for normalization
        :return: mean, std dev
        """
        # Load from cache if it exists
        dataset_specific_cache_file = ".cache/mpii_meanstd"
        if isfile(dataset_specific_cache_file):
            meanstd = torch.load(dataset_specific_cache_file)
            self.mean, self.std = meanstd["mean"], meanstd["stddev"]
            return self.mean, self.std

        mean = torch.zeros(3)
        std = torch.zeros(3)
        train_len = len(self.train)
        for index in self.train:
            if index % 100 == 0: print("In compute mean: At "+str(index)+" out of "+str(train_len))
            a = self.anno[index]
            img_path = os.path.join(self.img_folder, a['img_paths'])
            img = load_image(img_path) # CxHxW
            mean += img.view(img.size(0), -1).mean(1)
            std += img.view(img.size(0), -1).std(1)
        mean /= len(self.train)
        std /= len(self.train)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (mean[0], mean[1], mean[2]))
            print('    Std:  %.4f, %.4f, %.4f' % (std[0], std[1], std[2]))

        # Cache the computation (as it's slow)
        if not isdir(".cache"):
            mkdir_p(".cache")
        cache = {"mean": mean, "stddev": std}
        torch.save(cache, dataset_specific_cache_file)

        return mean, std


    def set_mean_stddev(self, mean, stddev):
        """
        Setter
        """
        self.mean = mean
        self.std = stddev


    def get_mean_stddev(self):
        """
        Getter
        """
        return self.mean, self.std


    def __getitem__(self, index):
        """
        Get the 'index'th item from the dataset. The item being (img, 2d pose, meta) triplet.
        """
        # Unpacking
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        # Get the original image path
        img_path = os.path.join(self.img_folder, a['img_paths'])

        # Load points, and their visibilities. a['joint_self'] is an array of [x,y, visible] of length 16
        pts = torch.Tensor(a['joint_self'])
        jnts_visible = pts[:,2]
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_numpy_image(img_path)  # CxHxW

        # If not, "no_random_masking" then randomly mask the image (mask may copy img data to provide "no masking")
        if self.add_random_masking:
            pts_coords = pts[:, :2]
            should_mask, (min_x, max_x, min_y, max_y), mask = \
                generate_random_mask(img, pts_coords, self.mask_prob, self.orientation_prob, self.mean_valued_prob,
                                     self.mean, self.max_cover_ratio, self.noise_std)
            if should_mask:
                img[:, min_x:max_x, min_y:max_y] = mask

        r = 0
        if self.augment_data:
            img = torch.Tensor(img)

            # Generate a random scale and rotation
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Convert numpy
        img = torch.Tensor(img)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i] = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'visible': jnts_visible,
                'filename': img_path}

        # Add headbox information to meta if we are in the validation dataset
        if not self.is_train:
            meta['headbox'] = a['headbox']

        return inp, target, meta


    def __len__(self):
        """
        Size of the dataset
        """
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
