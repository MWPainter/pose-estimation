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

idx = [1,2,3,4,5,6,11,12,15,16]


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
    model = load_model(model_file, options)
    dataset = run_model(model, data_input_dir, options)
    save_preds(dataset, data_output_dir)



def load_model(model_file, args):
    """
    Load the PyTorch stacked hourglass model

    :param model_file: The file for the saved model
    :param args: The arguments (or options) passed to the script. Defaults specify the architecture
    :return: A PyTorch nn.Module object for the trained Stacked Hourglass network
    """
    # Make the model
    model = models.__dict__['hg'](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # Load in weights from checkpoint + set in eval mode
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model



def run_model(model, data_input_dir, args):
    """
    Run a trained model on an entire dataset

    :param model: PyTorch nn.Module object for the trained Stacked Hourglass network
    :param data_input_dir: Directory for the dataset to run network on
    :param args: The arguments (or options) passed to the script. Defaults specify the architecture
    :return: PyTorch Dataset object of 2D pose predictions
    """
    # Make the dataset object (for now just hard coded dataset)
    mpii_dataset = datasets.Mpii('stacked_hourglass/data/mpii/mpii_annotations.json', 'stacked_hourglass/data/mpii/images',
                        sigma=args.sigma, label_type=args.label_type)

    # Placeholder dictionary for predictions
    predictions = {}

    # Loop through each item of the dataset
    for i in range(len(mpii_dataset)):
        # Progress
        if i % 100 == 0:
            print("At " + str(i) + " out of " + str(len(mpii_dataset)) + ".")

        # Get info about the file
        inputs, _, meta = mpii_dataset[i]
        filename = mpii_dataset.anno[mpii_dataset.train[i]]['img_paths']

        # Compute and store the prediction (unsqueeze input to make a batch size of one)
        input_var = torch.autograd.Variable(inputs.unsqueeze(0).cuda(), volatile=True)
        output = model(input_var)
        score_map = output[-1].data.cpu()
        joint_preds = final_preds(score_map, [meta['center']], [meta['scale']], [64, 64])

        # Squeeze output to remove the phantom batch size + put into dataset
        predictions[filename] = joint_preds.view(joint_preds.size()[1:])

    return predictions



def save_preds(dataset, data_output_dir):
    """
    Save the PyTorch Dataset of predictions to a file

    :param dataset: The PyTorch Dataset object of predictions
    :param data_output_dir: The filename for the file to save
    """
    # Make directory if it doesn't exists
    if not isdir(data_output_dir):
        mkdir_p(data_output_dir)

    # Just save the predictions in the correct place via PyTorch
    torch.save(dataset, data_output_dir+"/2dposes")



