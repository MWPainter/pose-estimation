from __future__ import print_function, absolute_import

import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
# import torchvision.datasets as datasets

from stacked_hourglass.pose import Bar
# from stacked_hourglass.pose.utils.logger import Logger, savefig
from stacked_hourglass.pose.utils.evaluation import accuracy, AverageMeter, final_preds
# from stacked_hourglass.pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from .osutils import mkdir_p, isfile, isdir, join
# from stacked_hourglass.pose.utils.imutils import batch_with_heatmap
# from stacked_hourglass.pose.utils.transforms import fliplr, flip_back
from .. import models
from .. import datasets



def run(options):
    """
    Run the model on a dataset and save it

    Important options params:
    options.load: The file for the saved model
    options.data_dir: The input directory for data (RGB images) to pass through the network
    options.output_dir: The directory to store output predictions

    :param options: The options passed in by command line
    """
    # Unpack options
    model_file = options.load
    data_input_dir = options.data_dir
    data_output_dir = options.output_dir

    # Run
    model, input_dataset = _load_model_and_dataset(model_file, data_input_dir, options)
    dataset, ground_truths = _run_model(model, input_dataset)
    _save_preds(dataset, ground_truths, data_output_dir)



def _load_model_and_dataset(model_file, data_input_dir, args):
    """
    Load the PyTorch stacked hourglass model

    :param model_file: The file for the saved model
    :param data_input_dir: Directory for the dataset to run network on
    :param args: The arguments (or options) passed to the script. Needed to specify the architecture
    :return: A PyTorch nn.Module object for the trained Stacked Hourglass network
        and the dataset object
    """
    # Make the model
    model = models.__dict__['hg'](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # Load in weights from checkpoint + set in eval mode
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # create the dataset, NOT in train mode, and load the mean and stddev (if not a pre-trained model)
    mean = checkpoint['mean'] if 'mean' in checkpoint else None
    stddev = checkpoint['stddev'] if 'stddev' in checkpoint else None
    dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json', 'stacked_hourglass/data/mpii/images',
                        sigma=args.sigma, label_type=args.label_type, augment_data=False, mean=mean, stddev=stddev)

    # if the model is pre-trained, then the mean/stddev caching was done differently (so cache it!)
    if 'mean' not in checkpoint or 'stddev' not in checkpoint:
        checkpoint['mean'], checkpoint['stddev'] = dataset.get_mean_stddev()
        torch.save(checkpoint, model_file)

    return model, dataset



def _run_model(model, dataset):
    """
    Run a trained model on an entire dataset

    :param model: PyTorch nn.Module object for the trained Stacked Hourglass network
    :param dataset: The input dataset to run the model on.
    :return: PyTorch Dataset object of 2D pose predictions
    """
    # Placeholder dictionarys for predictions and ground truths
    predictions = {}
    ground_truths = {}

    # Loop through each item of the dataset
    for i in range(len(dataset)):
        # Progress
        if i % 100 == 0:
            print("At " + str(i) + " out of " + str(len(dataset)) + ".")

        # Get info about the file
        inputs, targets, meta = dataset[i]
        filename = dataset.anno[dataset.train[i]]['img_paths']

        # Compute and store the prediction (unsqueeze input to make a batch size of one)
        input_var = torch.autograd.Variable(inputs.unsqueeze(0).cuda(), volatile=True)
        output = model(input_var)
        score_map = output[-1].data.cpu()
        joint_preds = final_preds(score_map, [meta['center']], [meta['scale']], [64, 64])

        # Squeeze output to remove the phantom batch size + put into dataset
        predictions[filename] = joint_preds.view(joint_preds.size()[1:])
        ground_truths[filename] = targets

    return predictions, ground_truths



def _save_preds(dataset, ground_truths, data_output_dir):
    """
    Save the PyTorch Dataset of predictions to a file

    :param dataset: The PyTorch Dataset object of predictions
    :param ground_truths: The ground truth 2D joint locations
    :param data_output_dir: The filename for the file to save
    """
    # Make directory if it doesn't exists
    if not isdir(data_output_dir):
        mkdir_p(data_output_dir)

    # Just save the predictions in the correct place via PyTorch
    torch.save(dataset, data_output_dir+"/2dposes")
    torch.save(ground_truths, data_output_dir+"/ground_truths")



