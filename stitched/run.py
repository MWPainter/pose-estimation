from __future__ import print_function, absolute_import

import os
import time

import torch

import stacked_hourglass.pose.datasets as datasets
from stitched.stitched_network import StitchedNetwork
from utils.transform import mpii_to_h36m_joints

from utils.osutils import mkdir_p, isfile, isdir, join



def run_mpii(options):
    """
    Run the model on the MPII dataset and save it

    Important options params:
    options.load: The file for the saved model
    options.data_dir: The input directory for data (RGB images) to pass through the network
    options.output_dir: The directory to store output predictions

    :param options: The options passed in by command line
    """
    # Unpack options
    hg_model_file = options.load_hourglass
    threed_baseline_model_file = options.load_2d3d
    data_input_dir = options.data_dir
    data_output_dir = options.output_dir

    # Run
    model, data_loader = load_model_and_dataset_mpii(hg_model_file, threed_baseline_model_file, data_input_dir, options)
    twod_predictions, threed_predictions = _run_model_mpii(model, data_loader)
    _save_preds(twod_predictions, threed_predictions, data_output_dir)



def load_model_and_dataset_mpii(hg_file, threed_baseline_file, data_input_dir, args):
    """
    Load the PyTorch stitched model

    :param hg_file: The file for the saved hourglass model
    :param threed_baseline_file: The file for the saved 3D baseline model
    :param data_input_dir: Directory for the dataset (of RGB images) to run network on
    :param args: (Ignored for now). The arguments (or options) passed to the script. Needed to specify the architecture
    :return: A PyTorch nn.Module object for a stitched network and a PyTorch dataloader object
    """
    # Load the hourglass checkpoint
    checkpoint = torch.load(hg_file)

    # create the dataset, NOT in train mode, and load the mean and stddev (if not a pre-trained model)
    mean = checkpoint['mean'] if 'mean' in checkpoint else None
    stddev = checkpoint['stddev'] if 'stddev' in checkpoint else None
    dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json', 'stacked_hourglass/data/mpii/images',
                        sigma=args.sigma, label_type=args.label_type, augment_data=False, train=args.run_with_train, mean=mean, stddev=stddev, args=args)

    # if the model is pre-trained, then the mean/stddev caching was done differently (so cache it!)
    if 'mean' not in checkpoint or 'stddev' not in checkpoint:
        checkpoint['mean'], checkpoint['stddev'] = dataset.get_mean_stddev()
        torch.save(checkpoint, hg_file)

    # Make the model and load weights from checkpoints and set to eval mode
    model = StitchedNetwork(hg_stacks=args.stacks, hg_blocks=args.blocks, hg_num_classes=args.num_classes,
                            hg_batch_norm_momentum=args.batch_norm_momentum, hg_use_layer_norm=args.use_layer_norm,
                            hg_mean=mean, hg_std=stddev, width=256, height=256, transformer_fn=mpii_to_h36m_joints,
                            dataset_normlization=args.dataset_normalization)
    model.load(hg_file, threed_baseline_file)
    model = model.cuda()
    model.eval()

    # wrap in a data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    return model, data_loader



def load_model_and_dataset_h36m(file, hg_file, threed_baseline_file, args):
    """
    Load the PyTorch stitched model.

    We assume that the network uses the color normalizationa dn

    :param file: File for the entire network
    :param hg_file:
    :param threed_baseline_file:
    :param args:
    :return:
    """
    # If file isn't empty, then we have a complete checkpoint. Which can be loaded by setting the following
    if file != '':
        hg_file = file
        threed_baseline_file = None

    # Get the mean and std (possibly from the MPII dataset, but cache it if we did)
    checkpoint = torch.load(hg_file)
    mean = checkpoint['mean'] if 'mean' in checkpoint else None
    stddev = checkpoint['stddev'] if 'stddev' in checkpoint else None
    if mean is None or stddev is None:
        dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json', 'stacked_hourglass/data/mpii/images',
                            sigma=args.sigma, label_type=args.label_type, augment_data=False, train=args.run_with_train, mean=mean, stddev=stddev, args=args)
        mean, stddev = dataset.get_mean_stddev()
        checkpoint['mean'] = mean
        checkpoint['stddev'] = stddev
        torch.save(checkpoint, hg_file)

    # Make the model and load weights from checkpoints and set to eval mode
    model = StitchedNetwork(hg_stacks=args.stacks, hg_blocks=args.blocks, hg_num_classes=args.num_classes,
                            hg_batch_norm_momentum=args.batch_norm_momentum, hg_use_layer_norm=args.use_layer_norm,
                            hg_mean=mean, hg_std=stddev, width=256, height=256, transformer_fn=mpii_to_h36m_joints)
    model.load(hg_file, threed_baseline_file)
    model = model.cuda()
    model.eval()

    # Make the dataset and dataloader, manually setting the mean and std
    dataset = Human36mDataset(dataset_path=data_input_dir, is_train=False,
                                  dataset_normalization=dataset_normalization, load_image_data=True)
    dataset.set_color_mean(model.hg_mean)
    dataset.set_color_std(model.hg_std)
    data_loader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    return model, data_loader




def _run_model_mpii(model, data_loader):
    """
    Run a trained model on an entire dataset

    :param model: PyTorch nn.Module object for the trained Stacked Hourglass network
    :param data_loader: A PyTorch DataLoader object for the dataset.
    :return: A 'dataset' map of 2D pose predictions and a 'dataset' map of 3D pose predictions
    """
    # Placeholder dictionarys for predictions and ground truths
    twod_predictions = {}
    threed_predictions = {}

    # Get the PyTorch tensors for dataset normalization in 3d baseline (not used if instance norm)
    dataset = Human36mDataset(dataset_path=data_input_dir, is_train=False,
                                  dataset_normalization=dataset_normalization, load_image_data=True)
    mean, std = torch.Tensor(dataset.pose_2d_mean), torch.Tensor(dataset.pose_2d_std)

    # Loop through each batch of the dataset
    for i, (inputs, target, meta) in enumerate(data_loader):
        # Progress
        if i % 100 == 0:
            print("At " + str(i) + " out of " + str(len(data_loader)) + ".")

        # add pose normalization stats to the meta
        meta['2d_mean'] = mean
        meta['2d_std'] = std

        # Compute and store the prediction (unsqueeze input to make a batch size of one)
        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        _, twod_preds, threed_preds = model(input_var, meta)
        # score_map = output[-1].data.cpu()
        # joint_preds = final_preds(score_map, [meta['center']], [meta['scale']], [64, 64])

        # Squeeze output to remove the phantom batch size + put into dataset
        dataset = data_loader.dataset
        index = meta['index']
        for j in range(inputs.size(0)):
            anno_index = dataset.train[index[j]] if dataset.is_train else dataset.valid[index[j]]
            filename = dataset.anno[anno_index]['img_paths']
            print(filename)
            quit()
            twod_predictions[filename] = twod_preds[j].cpu().detach()
            threed_predictions[filename] = threed_preds[j].cpu().detach()

    return twod_predictions, threed_predictions




def _run_model_h36m(model, data_loader):
    """
    Run a trained model on an entire dataset

    :param model: PyTorch nn.Module object for the trained Stacked Hourglass network
    :param data_loader: A PyTorch DataLoader object for the dataset.
    :return: A 'dataset' map of 2D pose predictions and a 'dataset' map of 3D pose predictions
    """
    # Placeholder dictionarys for predictions and ground truths
    twod_predictions = {}
    twod_ground_truths = {}
    threed_predictions = {}
    threed_ground_truths = {}
    metas = {}

    # Loop through each batch of the dataset
    for i, (inputs, _, _, targets, meta) in enumerate(data_loader):
        # Progress
        if i % 100 == 0:
            print("At " + str(i) + " out of " + str(len(data_loader)) + ".")

        # Compute and store the predictions
        meta['center'] = meta['img_center']
        meta['scale'] = meta['img_scale']
        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        _, twod_preds, threed_preds = model(input_var, meta)

        # Get the ground truths
        twod_gt = meta["2d_pose_orig_img"]
        threed_gt = targets

        # Put into dataset
        for j in range(inputs.size(0)):
            filename = meta["img_filename"][j]
            print(filename)
            quit()
            twod_predictions[filename] = twod_preds[j].cpu().detach()
            twod_ground_truths[filename] = twod_gt[j]
            threed_predictions[filename] = threed_preds[j].cpu().detach()
            threed_ground_truths[filename] = threed_gt[j]
            metas[filename] = meta

    return twod_predictions, threed_predictions, twod_ground_truths, threed_ground_truths, metas



def _save_preds(twod_predictions, threed_predictions, data_output_dir):
    """
    Save the PyTorch set of predictions to a file

    :param twod_predictions: The map object of 2D predictions that we wish to save
    :param threed_predictions: The map object of 3D predictions that we wish to save
    :param data_output_dir: The filename for the file to save
    """
    # Make directory if it doesn't exists
    if not isdir(data_output_dir):
        mkdir_p(data_output_dir)

    # Just save the predictions in the correct place via PyTorch
    torch.save(twod_predictions, data_output_dir+"/2dpreds")
    torch.save(threed_predictions, data_output_dir+"/3dpreds")



def run_h36m(options):
    """
    Run the model on the Human3.6m dataset and save it

    Important options params:
    options.load: The file for the entire saved model
    options.load_hourglass: The file for the saved hourglass model
    options.load_2d3d: The file for the 3D baseline network
    options.data_dir: The input directory for data (RGB images) to pass through the network
    options.output_dir: The directory to store output predictions

    :param options: The options passed in by command line
    """
    # Unpack options
    model_file = options.load
    hg_model_file = options.load_hourglass
    threed_baseline_model_file = options.load_2d3d
    data_output_dir = options.output_dir

    # Run
    model, data_loader = load_model_and_dataset_h36m(model_file, hg_model_file, threed_baseline_model_file, options)
    pred_2d, pred_3d, gt_2d, gt_3d = _run_model_h36m(model, data_loader)
    _save_preds_h36m(pred_2d, pred_3d, gt_2d, gt_3d, metas, data_output_dir)




def _save_preds(pred_2d, pred_3d, gt_2d, gt_3d, metas, data_output_dir):
    """
    TODO
    """
    # Make directory if it doesn't exists
    if not isdir(data_output_dir):
        mkdir_p(data_output_dir)

    # Just save the predictions in the correct place via PyTorch
    torch.save(pred_2d, data_output_dir + "/2dpreds")
    torch.save(pred_3d, data_output_dir + "/3dpreds")
    torch.save(gt_2d, data_output_dir + "/2dgt")
    torch.save(gt_3d, data_output_dir + "/3dgt")
    torch.save(metas, data_output_dir + "/metas")



