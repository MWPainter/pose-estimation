# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Import matplotlib and set backend to agg so it doesn't go wrong
# MUST be first, before ANY file imports numpy for example
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Relative imports
from stacked_hourglass.evaluation.utils import visualize as viz_2d_overlay
from stacked_hourglass.pose.utils.evaluation import final_preds
from stacked_hourglass.pose.utils.run import _load_model_and_dataset
from utils import data_utils
from utils.human36m_dataset import Human36mDataset
from utils.osutils import mkdir_p, isdir
from stacked_hourglass.pose.utils.transforms import color_denormalize
from twod_threed.src.viz import viz_2d_pose, viz_3d_pose
from twod_threed.src.datasets.human36m import get_3d_key_from_2d_key
from twod_threed.src.model import LinearModel


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
    total = len(twod_pose_preds)

    # Produce a visualization for each input image, outputting to 'output_dir' with the same image name as input
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            abs_filename = os.path.join(img_dir, filename)
            img = scipy.misc.imread(abs_filename)
            if not filename in twod_pose_preds:
                continue
            twod_overlay = viz_2d_overlay(img, twod_pose_preds[filename])
            threed_pose_viz = viz_3d_pose(threed_pose_preds[filename].numpy(), "data/2d3d_h36m")
            final_img = _pack_images([twod_overlay, threed_pose_viz])
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



def viz_orthog_transform(options):
    """
    Visual

    Options that should be included:
    options.data_dir: the directory for the dataset of poses
    options.load: checkpoint file for the model
    options.index: the index into the dataset to visualize
    options.num_orientations: the number of re-orientations to make
    options.dataset_normalization: if the network was trained with dataset normalizations
    options.output_dir: the directory to output the visualization image

    :param options: Options for the visualization, defined in options.py.
    """
    # Unpack options
    data_dir = options.data_dir
    model_checkpoint_file = options.load
    index = options.index
    num_orientations = options.num_orientations
    dataset_normalize = options.dataset_normalization

    # Make output dir
    if not isdir(options.output_dir):
        mkdir_p(options.output_dir)

    # Make the dataset object, and load the model, and put it in eval mode
    dataset = Human36mDataset(dataset_path=data_dir, orthogonal_data_augmentation=False,
                              z_rotations_only=options.z_rotations_only, dataset_normalization=dataset_normalize)
    model = LinearModel(dataset_normalized_input=dataset_normalize).cuda()
    ckpt = torch.load(model_checkpoint_file)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Loop
    vstack = []
    for i in range(num_orientations):
        # Get the data from the dataset
        _, _, pose_2d_gt, pose_3d_gt, meta = dataset[index]

        # Run the model to get the prediction (put it in a 'psuedo batch' of size 1)
        pose_3d_pred = model(torch.Tensor(pose_2d_gt).view((1,-1)).cuda()).cpu().detach().numpy()

        # Unnormalized poses (adding and remove the phantom batching as needed)
        pose_2d_gt = np.expand_dims(pose_2d_gt, axis=0)
        pose_3d_gt = np.expand_dims(pose_3d_gt, axis=0)
        pose_2d_gt_unnorm = data_utils.unNormalizeData(pose_2d_gt, meta, dataset_normalize, is_2d=True)[0]
        pose_3d_gt_unnorm = data_utils.unNormalizeData(pose_3d_gt, meta, dataset_normalize)[0]
        pose_3d_pred_unnorm = data_utils.unNormalizeData(pose_3d_pred, meta, dataset_normalize)[0]

        # Visualize in a hstack
        pose_2d_gt_img = viz_2d_pose(pose_2d_gt_unnorm)
        # pose_3d_gt_img = viz_3d_pose(pose_3d_gt_unnorm)
        print(meta['3d_pose_camera_coords'].shape)
        pose_3d_gt_img = viz_3d_pose(meta['3d_pose_camera_coords'])
        pose_3d_pred_img = viz_3d_pose(meta['3d_pose_camera_coords'])
        # pose_3d_pred_img = viz_3d_pose(pose_3d_pred_unnorm)


        # If it's the first iteration, now switch to using the orthogonal data augmentation
        if i == 0: dataset.orthogonal_data_augmentation = True

        # Append hstacked image to vstack
        vstack.append(_pack_images([pose_2d_gt_img, pose_3d_gt_img, pose_3d_pred_img]))

    # Compute the vstacked image and save it
    final_visualization = _pack_images_col(vstack)
    output_filename = os.path.join(options.output_dir, "{index}.jpg".format(index=index))
    scipy.misc.imsave(output_filename, final_visualization)



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
        for t in range(min(len(twod_pose_ground_truths[k2d]), 100)):
            twod_gt_viz = viz_2d_pose(twod_pose_ground_truths[k2d][t], options.data_dir)
            threed_gt_viz = viz_3d_pose(threed_pose_ground_truths[k3d][t], options.data_dir)
            threed_pred_viz = viz_3d_pose(threed_pose_preds[k2d][t].numpy(), options.data_dir)

            final_img = _pack_images([twod_gt_viz, threed_gt_viz, threed_pred_viz])
            scipy.misc.imsave(os.path.join(output_dir, str(k2d)+"_"+str(t)+".jpg"), final_img)

            # progress
            if i % 100 == 0:
                print("Visualized " + str(i) + " out of " + str(total))
            i += 1



def visualize_saliency_and_prob_maps(options, skeleton_overlay=False):
    """
    This is a visualization of 2D joint predictions.

    For each joint we will produce a row of images:
    Original Image, Joint Prediction, Saliency Map, Overlayed Saliency Map

    Then for each image, we will produce one row for each joint, at put them in a big collumn

    Options that should be included:
    options.img_dir: directory for the image(s)
    options.load: specifies the location of the saved (hourglass) model
    options.output_dir: specifies the location to save the (torch dataset of) pose predictions
    <any other model specific options you specified for training, e.g. --use_layer_norm, or --stacks 4>

    :param options: The options used to specify where to load images from and where to save the output etc
    :param skeleton_overlay: If we should overlay the image with a skeleton
    :return: Nothing
    """
    upasample_4x4 = torch.nn.Upsample(scale_factor=4)

    # if output directory doesnt exist, make it
    if not isdir(options.output_dir):
        mkdir_p(options.output_dir)

    # Load model (use helper from StackedHourglass' run.py)
    model, dataset = _load_model_and_dataset(options.load, options.img_dir, options)

    # Iterate through every image
    for i in range(len(dataset)):
        # Progress
        if i % 1 == 0:
            print("At " + str(i) + " out of " + str(len(dataset)) + ".")

        # Get the image, the ground truths and data about the image map
        inputs, targets, meta = dataset[i]
        filename = dataset.anno[dataset.train[i]]['img_paths']

        # Wrap input in a variable and set that it requires a gradient (so we can actually get gradient info)
        inputs_var = torch.autograd.Variable(inputs.unsqueeze(0))
        inputs_var.requires_grad_()

        # Run the model to get the output predictions
        output = model(inputs_var.cuda())
        score_map = output[-1].cpu().data
        joint_preds = final_preds(score_map, [meta['center']], [meta['scale']], [64, 64]).squeeze()

        # Compute the original image, as "inputs" is color normalized, so move from [-1,1] to [0,255]. Also transposes to go from [C,W,H] to [W,H,C], when there is a C dimension
        # If we have joint predictions, then overlay them also
        original_image = inputs.clone()
        color_denormalize(original_image, dataset.mean, dataset.std)
        original_image = original_image.numpy().transpose(1,2,0) * 255.0

        # If we have skeleton information, then, add the original image with skeleton overlay
        abs_filename = os.path.join(options.img_dir, filename)
        img = scipy.misc.imread(abs_filename)
        twod_overlay = viz_2d_overlay(img, joint_preds)

        # Compute the output from the network (which is a list and we only want the last, final set of scores). Output is of shape [1,num_joints,64,64] so squeeze and upsample
        scores = model(inputs_var.cuda())[-1].cpu().squeeze()
        scores_upsampled = upasample_4x4(scores.unsqueeze(0)).squeeze()

        # Saliency map is the gradient of the scores with respect to the scores. We want to do this one joint at a time
        packed_joint_imgs = []
        for joint in range(model.num_classes):
            # Comput saliency, i.e. gradient of the scores w.r.t input image
            joint_scores = scores[joint]
            if options.use_max_for_saliency_map:
                joint_scores_sum_or_max = torch.max(joint_scores)
            else:
                joint_scores_sum_or_max = torch.sum(joint_scores)
            joint_scores_sum_or_max.backward(retain_graph=True)
            saliency = torch.sum(inputs_var.grad.abs().squeeze(), dim=0)

            # Zero out any gradients (for next iteration)
            inputs_var.grad.zero_()
            model.zero_grad()

            # Create the images to stack. (prob dist is just [W,H] in shape, not [C,W,H])
            # Transpose probabilities and saliency maps, 'coz matplotlib seems to take y,x rather than x,y coords
            joint_prob_distr_image = _heatmap_from_prob_scores(scores_upsampled[joint].detach().numpy())
            saliency_image = _heatmap_from_prob_scores(saliency.numpy(), colormap=plt.cm.hot)
            saliency_overlay = _overlay_saliency(original_image, saliency.numpy())

            # Stack these images in a row
            imgs = []
            if skeleton_overlay and twod_overlay is not None: imgs.append(twod_overlay)
            imgs.extend([original_image, joint_prob_distr_image, saliency_image, saliency_overlay])
            packed_joint_imgs.append(_pack_images(imgs))

        # Stack images in a collumn and save
        final_visualization = _pack_images_col(packed_joint_imgs)
        output_filename = os.path.join(options.output_dir, filename)
        scipy.misc.imsave(output_filename, final_visualization)



def _softmax(x):
    """Simple numpy softmax function (because numpy doesn't have it...?)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def _heatmap_from_prob_scores(logits, colormap='RdBu_r'):
    """
    Return a heatmap of a probability distribution.

    :param logits: THe logits (inputs to a softmax function) to print the probability distribution of. A numpy array of shape [W,H]
    :return: A numpy array of shape [W,H,C] representing the heatmap image
    """
    # Compute the distribution
    probs = _softmax(logits)

    # Plot
    fig = plt.figure(figsize=(4.0, 4.0), dpi=64)
    plt.axis('off')
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.imshow(probs, cmap=colormap)
    #plt.pcolormesh(probs, cmap=colormap)
    #plt.colorbar()

    # convert fig to a numpy array
    # see: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # avoid unecessary memory consumption
    plt.close(fig)

    return data



def _overlay_saliency(image, saliency, lmda=0.25):
    """
    Overalying the saliency scores on a
    We do this by using softmax to turn the saliency scores into a probability distribution. Then, we highlight regions
    in the image by computing 1.0 * lambda + saliency * (1-lambda), for some lambda in (0,1).

    This gives each pixel a weight of lambda to 1.0 in the end image.

    :param image: The input image to the entire network, shape (W,H,C), values in range [0,255]
    :param saliency: The saliency map computed, shape (W,H), values in range [0,1]
    :return: The input image with regions of it highlighted according to the saliency map
    """
    # Compute the distribution, and then rescale so max value = 1.0
    # normalized_scores = _softmax(saliency)
    saliency_img = np.zeros(image.shape)
    normalized_saliency = saliency / np.max(saliency)
    saliency_img[:,:,0] = normalized_saliency
    return lmda * image + (1.0-lmda) * saliency_img * 255.0



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
        canvas[:height, x_running:x_running+width, :] = img
        x_running += width

    return canvas



def _pack_images_col(img_list):
    """
    Given a list of images, pack them into a single imge, putting them all into a collumn. (This is essentailly the
    flipped version of "pack_images")

    :param img_list: A list of images (as numpy arrays) which are to be concatenated into a single image
    :return:  A single image, containing each imge in 'img_list' as a sub image
    """
    # Comput the new shape
    y_total = 0
    x_max = 0
    for img in img_list:
        height, width, _ = img.shape
        y_total += height
        if x_max < width:
            x_max = width

    # Make a canvas and paste the image list into it
    canvas = np.zeros((y_total, x_max, 3))
    y_running = 0
    for img in img_list:
        height, width, _ = img.shape
        canvas[y_running:y_running+height, :width, :] = img
        y_running += height

    return canvas






if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # set random seeds
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)

    # run the appropriate 'script'
    if script == "2d_overlay_3d_pred":
        visualize_2d_overlay_3d_pred(options)
    elif script == "2d_overlay_3d_gt_3d_pred":
        visualize_2d_overlay_3d_gt_3d_pred(options)
    elif script == "2d_overlay":
        visualize_2d_overlay(options)
    elif script == "2d_gt_3d_gt_3d_pred":
        visualize_2d_pred_3d_gt_3d_pred(options)
    elif script == "saliency_and_prob_maps":
        visualize_saliency_and_prob_maps(options)
    elif script == "saliency_and_prob_maps_with_skeleton":
        visualize_saliency_and_prob_maps(options, True)
    elif script == "orthog_augmentation":
        viz_orthog_transform(options)







