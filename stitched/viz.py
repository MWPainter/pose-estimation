import cv2
import scipy
import os
import numpy as np
from twod_threed.src.viz import viz_3d_pose
from stacked_hourglass.evaluation.utils import visualize as viz_2d_overlay




def visualize_video(options):
    """
    Visualization run on a video. # TODO: Actually explain what it's doing
    :param options: # TODO: actually describe these
    """
    # Unpack optiosn
    video_filename = options.input_file
    file = options.load
    hg_file = options.load_hourglass
    threed_baseline_file = options.load_2d3d
    output_dir = options.output_dir

    # If file isn't empty, then we have a complete checkpoint. Which can be loaded by setting the following
    if file != '':
        hg_file = file
        threed_baseline_file = None

    # Load the model, and get the normalization statistics from the human3.6m dataset
    model, dataloader = load_model_and_dataset_h36m(file, hg_file, threed_baseline_file, args)
    pose_2d_mean = dataloader.dataset.pose_2d_mean
    pose_2d_std = dataloader.dataset.pose_2d_std

    # Load the video
    vid = cv2.VideoCapture(video_filename)

    # Loop through all frames of the video, and run model one frame at a time...
    i = 0
    succ, img = vid.read()
    while succ:
        # Progress
        if i % 100 == 0:
            print(i)

        # Get meta data
        height, width, channels = img.shape
        center = [height // 2, width // 2]
        scale = max(*center) / 256.0

        meta = {
            'center': torch.Tensor(center),
            'scale': torch.Tensor(scale),
            '2d_mean': pose_2d_mean,
            '2d_std': pose_2d_std,
        }

        # Forward pass
        img = np.transpose(img, (2,0,1))
        inputs = torch.Variable(torch.Tensor(img))

        _, twod_preds, threed_preds = model(inputs, meta)

        # Make visualization image
        twod_preds = twod_preds.numpy()
        threed_preds = threed_preds.numpy()
        twod_overlay = viz_2d_overlay(img, twod_preds)
        threed_viz = viz_3d_pose(threed_preds)
        out_img = _pack_images([twod_overlay, threed_viz])

        # Save the image
        output_filename = os.path.joint(output_dir, str(i))
        scipy.misc.imsave(output_filename, out_img)

        # Next image
        succ, img = vid.read()
        i += 1








def _pack_images(img_list):
    """
    Given a list of images, pack them into a single image, putting them all into a row.

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
        canvas[:height, x_running:x_running + width, :] = img
        x_running += width

    return canvas