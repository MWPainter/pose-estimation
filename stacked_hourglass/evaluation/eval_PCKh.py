from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
from utils import visualize
import numpy as np
import os
import sys


valid_preds_file = sys.argv[1] # 'checkpoint/mpii/' + model_name + '/preds_valid.mat', model_name='hg4'

def compute_PCKh_curve(pred_filename, modelname):
    """
    Print the PCKh@0.5 scores for a model and returns a vector of PCKh@x scores. (This was a script int he original
    version of this repo.

    Michael: Not touching this code other than wrapping it in a script...

    :param filename: THe filename of the predictions saved by the stacked hourglass network
    :return: PCKh, an array where PCKh[i] is the PCKh@(i*0.01) score
    """
    gt_preds_prt1_file = 'stacked_hourglass/evaluation/data/detections.mat' # sys.argv[2] # 'stacked_hourglass/evaluation/data/detections.mat'
    gt_preds_prt2_file = 'stacked_hourglass/evaluation/data/detections_our_format.mat' # sys.argv[3] # 'stacked_hourglassevaluation/data/detections_our_format.mat'

    detection = loadmat(gt_preds_prt1_file)
    det_idxs = detection['RELEASE_img_index']
    debug = 0
    threshold = 0.5
    SC_BIAS = 0.6

    pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

    dict = loadmat(gt_preds_prt2_file)
    dataset_joints = dict['dataset_joints']
    jnt_missing = dict['jnt_missing']
    pos_gt_src = dict['pos_gt_src']
    headboxes_src = dict['headboxes_src']

    #predictions
    predfile = pred_filename
    preds = loadmat(predfile)['preds']
    pos_pred_src = transpose(preds, [1, 2, 0])


    if debug:

        for i in range(len(det_idxs[0])):
            anno = mat['RELEASE']['annolist'][0, 0][0][det_idxs[0][i] - 1]
            fn = anno['image']['name'][0, 0][0]
            imagePath = '/data/mpii/images/' + fn
            oriImg = sio.imread(imagePath)
            pred = pos_pred_src[:, :, i]
            visualize(oriImg, pred, pa)


    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
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
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)


    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    name = predfile.split(os.sep)[-1]
    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    print("Model,  Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean")
    print('{:s}   {:.2f}  {:.2f}     {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(modelname, PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
            , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
            , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))

    # Compute all of the curves we want
    curves = {
        'Mean': np.mean(pckAll, axis=1),
        'Head': pckAll[:,head],
        'Shoulder': 0.5 * (pckAll[:,lsho] + pckAll[:,rsho]),
        'Elbow': 0.5 * (pckAll[:,lelb] + pckAll[:,relb]),
        'Wrist': 0.5 * (pckAll[:,lwri] + pckAll[:,rwri]),
        'Hip': 0.5 * (pckAll[:,lhip] + pckAll[:,rhip]),
        'Knee': 0.5 * (pckAll[:,lkne] + pckAll[:,rkne]),
        'Ankle': 0.5 * (pckAll[:,lank] + pckAll[:,rank]),
    }

    return curves