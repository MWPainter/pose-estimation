from __future__ import absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy_PCK', 'accuracy_PCKh']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def accuracy_PCK(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1], thr=thr)
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc

def accuracy_PCKh(output, target, meta, idxs, joint_name_to_idx, threshold=0.5):
    # Constants
    SC_BIAS = 0.6

    # Mash data into numpy arrays of the appropriate size and shape
    # The code for computing PCKh (for some reason), uses batch/dataset size in the last dimension
    # Therefore we require the following shapes:
    # joints_visible.shape = (16, N)
    # headboxs = (2,2,N).                   (this means it's something like the top left + bottom right corners)
    # preds = (16,2,N)
    # All of the transposing is just to shift the batch size to the correct dimension
    preds = final_preds(output, meta['center'], meta['scale'], [64,64])
    pos_pred_src = np.transpose(preds.numpy(), [1, 2, 0])

    pos_gt_np = meta['pts'][:,:,:2]
    pos_gt_src = np.transpose(pos_gt_np.numpy(), [1,2,0])

    jnt_visible = np.transpose(meta['visible'].numpy(), [1,0])
    headboxes_src = np.transpose(meta['headbox'].numpy(), [1,2,0])

    # Compute PCKh score, taken from eval_PCKh.py script
    #jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    jnt_count_fake = np.maximum(1.0, jnt_count) # hack, however, if jnt_count[i]=0, then likely less_than_threshold[i]=0, and want to treat 0/0 as 0 here
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count_fake)
    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    # Return the mean PCKh and the PCKh per joint
    head = joint_name_to_idx['head']
    lsho = joint_name_to_idx['lsho']
    lelb = joint_name_to_idx['lelb']
    lwri = joint_name_to_idx['lwri']
    lhip = joint_name_to_idx['lhip']
    lkne = joint_name_to_idx['lkne']
    lank = joint_name_to_idx['lank']
    rsho = joint_name_to_idx['rsho']
    relb = joint_name_to_idx['relb']
    rwri = joint_name_to_idx['rwri']
    rhip = joint_name_to_idx['rhip']
    rkne = joint_name_to_idx['rkne']
    rank = joint_name_to_idx['rank']

    return np.mean(PCKh), {
        'head':     (PCKh[head], jnt_count[head]),
        'shoulder': (0.5 * (PCKh[lsho] + PCKh[rsho]), jnt_count[lsho] + jnt_count[rsho]),
        'elbow':    (0.5 * (PCKh[lelb] + PCKh[relb]), jnt_count[lelb] + jnt_count[relb]),
        'wrist':    (0.5 * (PCKh[lwri] + PCKh[rwri]), jnt_count[lwri] + jnt_count[rwri]),
        'hip':      (0.5 * (PCKh[lhip] + PCKh[rhip]), jnt_count[lhip] + jnt_count[rhip]),
        'knee':     (0.5 * (PCKh[lkne] + PCKh[rkne]), jnt_count[lkne] + jnt_count[rkne]),
        'ankle':    (0.5 * (PCKh[lank] + PCKh[rank]), jnt_count[lank] + jnt_count[rank]),
    }


def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type
    return final_preds_post_processing(output, coords, center, scale, res)

def final_preds_post_processing(output, coords, center, scale, res, cuda=False):
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                if cuda: diff = diff.cuda()
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds




    

