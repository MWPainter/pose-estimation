import torch
import numpy as np

from utils import data_utils




def print_avg_joint_err(options):
    """
    Given 3D ground truth predictions and 3D predictions, in dictionaries where the keys correspond,
    compute the average joint error.

    Options:
    options.threed_pose_ground_truths: a PyTorch file containing 3D pose ground truths.
    options.threed_pose_estimations: a PyTorch file containing 3D pose estimations.
    options.metas: a PyTorch file containing all of the meta data for each example

    :param options:
    :return:
    """
    # TODO: compute this PER JOINT and PER ACTION. And compute a nice lil grid...
    # Load values
    preds = torch.load(options.threed_pose_estimations)
    gts = torch.load(options.threed_pose_ground_truths)
    metas = torch.load(options.metas)
    dataset_normalization = options.dataset_normalization

    # For each prediction, add an error
    errs = []
    for filename in pred:
        errs.append(data_utils.compute_3d_pose_error_distances(preds[filename], gts[filename], meta[filename],
                                                               dataset_normalization=dataset_normalization,
                                                               procrustes=False))

    # Print the mean value
    print("Mean joint error of these predictions is: {avg_err}".format(avg_err=np.mean(errs)))