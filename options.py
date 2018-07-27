#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint



__all__ = ['Options']



actions = ["all",
           "All",
           "Directions",
           "Discussion",
           "Eating",
           "Greeting",
           "Phoning",
           "Photo",
           "Posing",
           "Purchases",
           "Sitting",
           "SittingDown",
           "Smoking",
           "Waiting",
           "WalkDog",
           "Walking",
           "WalkTogether"]



# Defaults dictionary to provide different default arguments for different scripts
defaults = \
    {
        "2d3d":
            {
                # General options
                "data_dir": "data/2d3d_h36m",
                "checkpoint_dir": "model_checkpoints",
                "output_dir": "data/2d3d_output",

                # Training options
                "epochs": 200,
                "lr": 1.0e-3,
                "train_batch_size": 64,
                "test_batch_size": 64,
            },

        "hourglass":
            {
                # General options
                "data_dir": "data/h36m",
                "checkpoint_dir": "model_checkpoints",
                "output_dir": "data/hourglass_output",

                # Training options
                "epochs": 90,
                "lr": 2.5e-4,
                "train_batch_size": 6,
                "test_batch_size": 6,
            },

        "":
            {
                # General options
                "data_dir": "data",
                "checkpoint_dir": "model_checkpoints",
                "output_dir": "data",

                # Training options
                "epochs": 50,
                "lr": 1.0e-3,
                "train_batch_size": 6,
                "test_batch_size": 6,
            },
    }



class Options:
    def __init__(self, script_id):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        self.script_id = script_id if script_id is not None else ""


    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir',       type=str, default=defaults[self.script_id]["data_dir"], help='path to dataset')
        self.parser.add_argument('--checkpoint_dir', type=str, default=defaults[self.script_id]["checkpoint_dir"], help='path to store model checkpoints')
        self.parser.add_argument('--output_dir',     type=str, default=defaults[self.script_id]["output_dir"], help='where to store output (if any)')

        self.parser.add_argument('--exp',            type=str, default='0', help='ID of experiment')
        #self.parser.add_argument('--ckpt',           type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--load',           type=str, default='', help='path to load a pretrained checkpoint')

        self.parser.add_argument('--test',           dest='test', action='store_true', help='test')
        self.parser.add_argument('--resume',         dest='resume', action='store_true', help='resume to train')

        self.parser.add_argument('--action',         type=str, default='All', choices=actions, help='All for all actions')

        # ===============================================================
        #                     General training options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=defaults[self.script_id]["lr"])
        self.parser.add_argument('--lr_decay', type=int, default=100000, help='# steps of lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--epochs', type=int, default=defaults[self.script_id]["epochs"])
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch_size', type=int, default=defaults[self.script_id]['train_batch_size'])
        self.parser.add_argument('--test_batch_size', type=int, default=defaults[self.script_id]['test_batch_size'])

        # ===============================================================
        #                     Hourglass model options
        # ===============================================================
        self.parser.add_argument('--stacks', default=8, type=int, metavar='N',
                            help='Number of hourglasses to stack')
        self.parser.add_argument('--features', default=256, type=int, metavar='N',
                            help='Number of features in the hourglass')
        self.parser.add_argument('--blocks', default=1, type=int, metavar='N',
                            help='Number of residual modules at each location in the hourglass')
        self.parser.add_argument('--num-classes', default=16, type=int, metavar='N',
                            help='Number of keypoints')


        # ===============================================================
        #                     Hourglass training options
        # ===============================================================
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self.parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        self.parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        self.parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                            help='show intermediate results')

        # ===============================================================
        #                     Hourglass data processing options
        # ===============================================================
        self.parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                            help='flip the input during validation')
        self.parser.add_argument('--sigma', type=float, default=1,
                            help='Groundtruth Gaussian sigma.')
        self.parser.add_argument('--sigma-decay', type=float, default=0,
                            help='Sigma decay rate for each epoch.')
        self.parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                            choices=['Gaussian', 'Cauchy'],
                            help='Labelmap dist type: (default=Gaussian)')
        self.parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                            help='Decrease learning rate at these epochs.')
        self.parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')

        # ===============================================================
        #                     2D3D model options
        # ===============================================================
        self.parser.add_argument('--max_norm',       dest='max_norm', action='store_true', help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size',    type=int, default=1024, help='size of each model layer')
        self.parser.add_argument('--num_stage',      type=int, default=2, help='# layers in linear model')

        # ===============================================================
        #                     2D3D training options
        # ===============================================================
        self.parser.add_argument('--use_hg',         dest='use_hg', action='store_true', help='whether use 2d pose from hourglass')
        self.parser.add_argument('--job',            type=int,    default=8, help='# subprocesses to use for data loading')
        self.parser.add_argument('--no_max',         dest='max_norm', action='store_false', help='if use max_norm clip on grad')
        self.parser.add_argument('--max',            dest='max_norm', action='store_true', help='if use max_norm clip on grad')
        self.parser.set_defaults(max_norm=True)
        self.parser.add_argument('--procrustes',     dest='procrustes', action='store_true', help='use procrustes analysis at testing')

        # ===============================================================
        #                     "Stitched" training options
        # ===============================================================
        self.parser.add_argument('--load_hourglass',  type=str, help='Checkpoint file for a pre-trained hourglass model.')
        self.parser.add_argument('--load_2d3d',       type=str, help='Checkpoint file for a pre-trained 2d3d model')


    def _print(self):
        # Pretty prints the options we parses
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")


    def parse(self):
        # Parse the (known) arguments (with respect to the parser)
        self._initial()
        self.opt, _ = self.parser.parse_known_args()

        # Perform some validation on input
        checkpoint_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if self.opt.load:
            if not os.path.isfile(self.opt.load):
                print ("{} is not found".format(self.opt.load))

        # Set internal variables
        self.opt.is_train = False if self.opt.test else True
        self.opt.checkpoint_dir = checkpoint_dir

        # PPrint options parsed (sanity check for user)
        self._print()
        return self.opt
