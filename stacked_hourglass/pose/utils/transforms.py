from __future__ import absolute_import

import os
import numpy as np
import random
import scipy.misc
import matplotlib.pyplot as plt
import torch

from .misc import *
from .imutils import *


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords


def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img



def generate_random_mask(img, pts, mask_prob, orientation_prob, mean_valued_prob, mean_values,
                         max_cover_ratio, noise_std):
    """
    Generate a random mask for a given image, with some probabilistic probabilities for the mask.
    We produce a bar that covers the entier width or height of an image. We align it such that the mask
    covers some of the pose estimates, but not all of it

    :param img: An image of shape (C, W, H) to generate a random mask for
    :param pts: The (ground truth) joint location for a person in the img 'img'
    :param mask_prob: The probability that the mask will actually change the image.
        To "not mask", we provide a "mask" which is identical to the image
    :param orientation_prob: Probability that the mask will be a vertical bar
    :param mean_valued_prob: Probability that the mask will be filled with the 'mean value' (rather than gaussian noise)
    :param max_cover_ratio: the maximum ratio of the bounding box (of the joint positions) the we allow to be covered
    :param noise_std: The stddev of the gaussian noise added, if the mask is filled with Gaussian noise
    :return: shouldMask, (x_min, x_max, y_min, y_max), mask. A mask with its bounding box and boolean indicating
        if we should mask at all. If we return False for "shouldMask" then the caller should ignore the rest of the
        output. The mask tensor's shape is (C, x_max-x_min, y_max-y_min).
    """
    # Here bounding box means all points p are minx <= p.x < maxx
    C, W, H = img.size()
    joint_x_min, joint_x_max, joint_y_min, joint_y_max = bounding_box(pts)

    # decide whether to mask
    pr = random.random()
    if pr > mask_prob:
        return False, (0, 0, 0, 0), None

    # decide how large the mask is (and if it's a horizontal or vertical bar)
    # be careful to make sure that the min_x isn't at the end of the image, and that width is always >= 1
    mask_x_min, mask_x_max, mask_y_min, mask_y_min = 0, 0, 0, 0
    pr = random.random()
    if pr < orientation_prob:
        mask_x_min = int(joint_x_min + (joint_x_max - joint_x_min - 1) * random.random())
        max_width = min((joint_x_max-joint_x_min)*max_cover_ratio, W - mask_x_min)
        mask_x_max = int(mask_x_min + max_width * random.random())
        mask_x_max = max(mask_x_max, mask_x_min + 1)
        mask_y_min = 0
        mask_y_max = H
    else:
        mask_x_min = 0
        mask_x_max = W
        mask_y_min = int(joint_y_min + (joint_y_max - joint_y_min - 1) * random.random())
        max_height = min((joint_y_max-joint_y_min)*max_cover_ratio, H - mask_y_min)
        mask_y_max = int(mask_y_min + max_height * random.random())
        mask_y_max = max(mask_y_max, mask_y_min+1)

    # Create a mask, filled with the mean value or with gaussian noise (centered around the mean, and clipped to (-1,1))
    # Had to be funky and make a (C,W,H) mean tensor out of a (C,) mean tensor
    bbox = (mask_x_min, mask_x_max, mask_y_min, mask_y_max)
    mask_width = mask_x_max-mask_x_min
    mask_height = mask_y_max-mask_y_min

    mean = mean_values.repeat((mask_width, mask_height, 1)).permute(2,0,1)
    pr = random.random()
    if pr < mean_valued_prob:
        mask = mean
    else:
        mask = torch.normal(mean=mean, std=noise_std)
    return True, bbox, mask



def bounding_box(pts):
    """
    Given some joint positions, return their bounding box

    :param pts: A 16x2 dimension tensor, or 32 dimension tensor that define the joint locations
    :return: A bounding box around the 2D coords, (x_min, x_max, y_min, y_max)
    """
    pts = pts.view(16,2)

    x_min = torch.min(pts[:,1])
    x_max = torch.max(pts[:,1])
    y_min = torch.min(pts[:,0])
    y_max = torch.max(pts[:,0])

    return int(x_min.item()), int(x_max.item()), int(y_min.item()), int(y_max.item())