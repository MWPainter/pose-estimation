# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Import matplotlib and set backend to agg so it doesn't go wrong
# MUST be first, before ANY file imports numpy for example
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Relative imports
from stacked_hourglass.evaluation.utils import visualize as viz_2d_overlay
from stacked_hourglass.pose.utils.osutils import mkdir_p, isdir
from twod_threed.src.viz import viz_2d_pose, viz_3d_pose
from twod_threed.src.datasets.human36m import get_3d_key_from_2d_key

# Absolute imports
import numpy as np
import torch
from options import Options
import os
import scipy
import sys


def visualize_2d_overlay_3d_pred(options):
    """
    Unpacks options and makes visualizations for 2d and 3d predictions.

    Images in the output from left to right are:
    1. Original image with 2D pose overlayed
    2. 3D prediction visualization

    Options that should be included:
    options.img_dir: the directory for the image
    options.twod_pose_estimations: a PyTorch file containing 2D pose estimations. Assumed to be a dict keyed by filenames
    options.threed_pose_estimations: a PyTorch file containing the 3D pose estimations. Assumed to be a dict keyed by filenames
    options.output_dir: a directory to output each visualization to

    :param options: Options for the visualizations, defined in options.py. (Including defaults).
    """
    # Load the predictions and unpack options
    img_dir = options.img_dir
    twod_pose_preds = torch.load(options.twod_pose_estimations)
    threed_pose_preds = torch.load(options.threed_pose_estimations)
    output_dir = options.output_dir

    # Make dir for output if it doesnt exist
    if not isdir(output_dir):
        mkdir_p(output_dir)

    i = 0
    total = len(os.listdir(img_dir))

    # Produce a visualization for each input image, outputting to 'output_dir' with the same image name as input
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            abs_filename = os.path.join(img_dir, filename)
            img = scipy.misc.imread(abs_filename)
            if not filename in twod_pose_preds:
                continue
            twod_overlay = viz_2d_overlay(img, twod_pose_preds[filename])
            threed_pose_viz = viz_3d_pose(threed_pose_preds[filename])
            final_img = pack_images([twod_overlay, threed_pose_viz])
            scipy.misc.imsave(os.path.join(output_dir, filename), final_img)

            # progress
            if i % 100 == 0:
                print("Visualized " + str(i) + " out of " + str(total))
            i += 1


def visualize_2d_overlay(options):
    """
    Unpacks options and makes visualizations for 2d and 3d predictions.

    Images in the output from left to right are:
    1. Original image with 2D pose overlayed
    2. 3D prediction visualization

    Options that should be included:
    options.img_dir: the directory for the image
    options.twod_pose_estimations: a PyTorch file containing 2D pose estimations. Assumed to be a dict keyed by filenames
    options.output_dir: a directory to output each visualization to

    :param options: Options for the visualizations, defined in options.py. (Including defaults).
    """
    # Load the predictions and unpack options
    img_dir = options.img_dir
    twod_pose_preds = torch.load(options.twod_pose_estimations)
    output_dir = options.output_dir

    # Make dir for output if it doesnt exist
    if not isdir(output_dir):
        mkdir_p(output_dir)

    i = 0
    total = len(os.listdir(img_dir))

    # Produce a visualization for each input image, outputting to 'output_dir' with the same image name as input
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # print(filename)
            abs_filename = os.path.join(img_dir, filename)
            if not filename in twod_pose_preds:
                continue
            img = scipy.misc.imread(abs_filename)
            twod_overlay = viz_2d_overlay(img, twod_pose_preds[filename])
            scipy.misc.imsave(os.path.join(output_dir, filename), twod_overlay)

            # progress
            if i % 100 == 0:
                print("Visualized " + str(i) + " out of " + str(total))
            i += 1


def visualize_2d_overlay_3d_gt_3d_pred(options):
    """
    Same as visualize_2d_and_3d, but adds a ground truth visualization also.

    Images in the output from left to right are:
    1. original image with 2D pose overlayed
    2. 3D ground truth
    3. 3D prediction visualization

    Options that should be included:
    options.img_dir: the directory for the image
    options.twod_pose_estimations: a PyTorch file containing 2D pose estimations. Assumes the format of a dict,
        keyed by filenames
    options.threed_pose_ground_truths: a PyTorch file containing 3D pose ground truths
    options.threed_pose_estimations: a PyTorch file containing the 3D pose estimations. Assumes the format of a dict,
        keyed by filenames
    options.output_dir: a directory to output each visualization to

    :param options: Options for the visualizations, defined in options.py. (Including defaults).
    """
    raise NotImplementedError()



def visualize_2d_pred_3d_gt_3d_pred(options):
    """
    Visualize the 2D and 3D pose estimations on matplotlib axes. This is just an interface for twod_threed's
    visualizations

    Options that should be included:
    options.twod_pose_ground_truths: a PyTorch file containing 2D pose ground truths.
    options.threed_pose_ground_truths: a PyTorch file containing 3D pose ground truths.
    options.threed_pose_estimations: a PyTorch file containing 3D pose estimations.
    options.output_dir: A directory to output each visualization to

    :param options: Options for the visualizations, defined in options.py. (Including defaults).
    """
    # Unpack options
    twod_pose_ground_truths = torch.load(options.twod_pose_ground_truths)
    threed_pose_ground_truths = torch.load(options.threed_pose_ground_truths)
    threed_pose_preds = torch.load(options.threed_pose_estimations)
    output_dir = options.output_dir

    # Make dir for output if it doesnt exist
    if not isdir(output_dir):
        mkdir_p(output_dir)

    i = 0
    total = len(twod_pose_ground_truths)

    # Loop through each pose (each item in the dict is an array (in time) of 2d poses
    for k2d in twod_pose_ground_truths:
        k3d = get_3d_key_from_2d_key(k2d)
        for t in range(len(twod_pose_ground_truths[k2d])):
            twod_gt_viz = viz_2d_pose(twod_pose_ground_truths[k2d][t])
            threed_gt_viz = viz_3d_pose(threed_pose_ground_truths[k3d][t])
            threed_pred_viz = viz_3d_pose(threed_pose_preds[k2d][t])

            final_img = pack_images([twod_gt_viz, threed_gt_viz, threed_pred_viz])
            scipy.misc.imsave(os.path.join(output_dir, str(k2d)+"_"+str(t)+".jpg"), final_img)

            # progress
            if i % 100 == 0:
                print("Visualized " + str(i) + " out of " + str(total))
            i += 1


def pack_images(img_list):
    """
    Given a list of images, pack them into a single image.

    :param img_list: A list of images (as numpy arrays), which are to be concatenated into a single image
    :return: A single image, containing each image in 'img_list' as a subimage
    """
    # Compute the shape of the new visualization image
    x_total = 0
    y_max = 0
    for img in img_list:
        height, width, _ = img.shape
        x_total += width
        if y_max < height:
            y_max = height

    # Make a canvas and paste the image list into it
    canvas = np.zeros((y_max, x_total, 3))
    x_running = 0
    for img in img_list:
        height, width, _ = img.shape
        canvas[:height, x_running:x_running+width, :] = img
        x_running += width

    return canvas




if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "2d_overlay_3d_pred":
        visualize_2d_overlay_3d_pred(options)
    elif script == "2d_overlay_3d_gt_3d_pred":
        visualize_2d_overlay_3d_gt_3d_pred(options)
    elif script == "2d_overlay":
        visualize_2d_overlay(options)
    elif script == "2d_gt_3d_gt_3d_pred":
        visualize_2d_pred_3d_gt_3d_pred(options)
