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
# (This can be quite messy)
training_defaults = \
    {
        # for train/run
        "2d3d_h36m":
            {
                # General options
                "data_dir": "/data/h36m_pose",#"/var/storage/shared/pnrsy/mipain/Human3.6M/h36m_h5",#"/data/h36m_pose",
                "checkpoint_dir": "model_checkpoints",
                "output_dir": "",
                "tb_dir": "tb_logs/",

                # Training options
                "epochs": 500,
                "lr": 1.0e-3,
                "train_batch_size": 64,
                "test_batch_size": 64,
            },

        # for train/run
        "hourglass_mpii":
            {
                # General options
                "data_dir": "",
                "checkpoint_dir": "model_checkpoints",
                "output_dir": "",
                "tb_dir": "tb_logs/",

                # Training options
                "epochs": 200,
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
                "tb_dir": "tb_logs/",

                # Training options
                "epochs": 50,
                "lr": 1.0e-3,
                "train_batch_size": 6,
                "test_batch_size": 6,
            },
    }

visualize_defaults = \
    {

        "2d_overlay_3d_pred":
            {
                # General options
                "img_dir": "data/h36m",
                "twod_pose_estimations": "data/hourglass_2d_pred",
                "threed_pose_estimations": "data/twod_threed_3d_pred",
                "output_dir": "visualizations/2d_overlay_with_3d_pred",
            },
        "orthog_augmentation:":
            {
                "data_dir": "/data/h36m_pose",
            },
        "":
            {
                "img_dir": "",
                "twod_pose_estimations": "",
                "threed_pose_estimations": "",
                "output_dir": "",
            },
    }

class Options:
    def __init__(self, script_id):
        self._parser = argparse.ArgumentParser()
        self._opt = None
        self._script_id = script_id if script_id is not None else ""


    def _initial(self, t_defaults, v_defaults):
        """
        Initialize the argparse parser

        :param t_defaults: Dictionary of defaults for training params
        :param v_defaults: Dictionary of defaults for visualizations params
        """
        # ===============================================================
        #                     General options
        # ===============================================================
        self._parser.add_argument('--data_dir',       type=str, default=t_defaults["data_dir"], help='path to dataset')
        self._parser.add_argument('--checkpoint_dir', type=str, default=t_defaults["checkpoint_dir"], help='path to store model checkpoints')
        self._parser.add_argument('--output_dir',     type=str, default=t_defaults["output_dir"], help='where to store output (if any)')

        self._parser.add_argument('--exp',            type=str, default='0', help='ID of experiment')
        self._parser.add_argument('--load',           type=str, default='', help='path to load a pretrained checkpoint')

        self._parser.add_argument('--test',           dest='test', action='store_true', help='test')
        self._parser.add_argument('--resume',         dest='resume', action='store_true', help='resume to train')

        self._parser.add_argument('--action',         type=str, default='All', choices=actions, help='All for all actions')

        # ===============================================================
        #                     General training options
        # ===============================================================
        self._parser.add_argument('--lr', type=float, default=t_defaults["lr"])
        self._parser.add_argument('--lr_decay', type=int, default=100000, help='# steps of lr decay')
        self._parser.add_argument('--lr_gamma', type=float, default=1.0)
        self._parser.add_argument('--epochs', type=int, default=t_defaults["epochs"])
        self._parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')

        self._parser.add_argument('--train_batch_size', type=int, default=t_defaults['train_batch_size'])
        self._parser.add_argument('--test_batch_size', type=int, default=t_defaults['test_batch_size'])

        self._parser.add_argument('--no_grad_clipping', action='store_true', help='Option to turn off gradient clipping if need be')
        self._parser.add_argument('--grad_clip', type=float, default=10.0, help='Value to clip gradients to') # TODO: change this back to 0.5 if we can?

        self._parser.add_argument('--tb_dir', type=str, default=t_defaults["tb_dir"], help="Directory to write tensorboardX summaries.")
        self._parser.add_argument('--tb_log_freq', type=int, default=101, help='How frequently to update tensorboard summaries (num of iters per update). Default is prime incase we are computing different losses on different iterations.')

        self._parser.add_argument('--use_horovod', action='store_true', help='Use to specify if horovod should be used to train on multiple GPUs concurrently.')

        self._parser.add_argument('--seed', type=int, default=234, help='Specify a seed for random generation (math/numpy/PyTorch).')

        # ===============================================================
        #                     run.py specific options
        # ===============================================================
        self._parser.add_argument('--process_as_video', dest='process_as_video', action='store_true',
                                 help='Process videos when using run.py with a network that operates on single frames')
        self._parser.add_argument('--run_with_train', action='store_true', help='If we want to run/visualize using the training set rather than the validation set.')

        # ===============================================================
        #                     viz.py specific options
        # ===============================================================
        self._parser.add_argument('--img_dir',                 type=str, default=v_defaults["img_dir"], help='Directory for original images processed')
        self._parser.add_argument('--2d_pose_ground_truths', '--twod_pose_ground_truths',    dest='twod_pose_ground_truths', type=str, help='File containing the 2d ground truth poses for the images')
        self._parser.add_argument('--2d_pose_estimations', '--twod_pose_estimations',    dest='twod_pose_estimations', type=str, default=v_defaults["twod_pose_estimations"], help='File containing the 2d pose estimations for the images')
        self._parser.add_argument('--3d_pose_ground_truths', '--threed_pose_ground_truths',    dest='threed_pose_ground_truths', type=str, help='File containing the 3d ground truth poses for the images')
        self._parser.add_argument('--3d_pose_estimations', '--threed_pose_estimations',    dest='threed_pose_estimations', type=str, default=v_defaults["threed_pose_estimations"], help='File containing the 3d pose estimations for the images')

        self._parser.add_argument('--use_max_for_saliency_map', '--max_for_saliency', action="store_true", help="If we should use the max value from the prob scores (rather than a sum of values) when computing the saliency map.")

        self._parser.add_argument('--index', type=int, default=0, help='index into the dataset to use (orthogonal vizualization)')
        self._parser.add_argument('--num_orientations', type=int, default=16, help='the number of orientations to use in the orthogonal visualization')


        # ===============================================================
        #                     eval.py specific options
        # ===============================================================
        self._parser.add_argument('--prediction_files', type=str, nargs='+', default=[], help='A comma seperated list of filenames for "predictions", output by stacked hourglass models.')
        self._parser.add_argument('--model_names', type=str, nargs='+', default=[], help='A comma seperated list of model names, corresponding to the models used to produce the predictions.')
        self._parser.add_argument('--output_filename', type=str, default=None, help='An output filename to save the graph to')

        # ===============================================================
        #                     Hourglass model options
        # ===============================================================
        self._parser.add_argument('--stacks', default=8, type=int, metavar='N',
                            help='Number of hourglasses to stack')
        self._parser.add_argument('--features', default=256, type=int, metavar='N',
                            help='Number of features in the hourglass')
        self._parser.add_argument('--blocks', default=1, type=int, metavar='N',
                            help='Number of residual modules at each location in the hourglass')
        self._parser.add_argument('--num-classes', default=16, type=int, metavar='N',
                            help='Number of keypoints')
        self._parser.add_argument('--remove_intermediate_supervision', action='store_true', help='Remove supervision at the intermediate stages of stacked hourglass modules.')
        self._parser.add_argument('--add_attention', action='store_true', help='If we want to use time information via attention mechanism.')
        self._parser.add_argument('--scale_weight_factor', type=float, default=1.0, help='If we want to be able to scale all of the (trainable) weights by some constant factor at the start of training.')
        self._parser.add_argument('--use_layer_norm', action="store_true", help='Option to replace batchnorm by layernorm layers')
        self._parser.add_argument('--no_batch_norm_affine', action="store_true", help="Option to remove the affine parameters in batch norm layers in the Hourglass module")

        # joint visibility
        self._parser.add_argument('--predict_joint_visibility', action="store_true", help="If we want to add joint prediction into the training of the stacked hourglass")
        self._parser.add_argument('--joint_visibility_loss_coeff', type=float, default=1e-4, help="The coefficient to use for the joint visibility in the loss function, if predicting joint visibilities.")

        # ===============================================================
        #                     Hourglass training options
        # ===============================================================
        self._parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self._parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        self._parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        self._parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        self._parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                            help='show intermediate results')
        self._parser.add_argument('--schedule', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
        self._parser.add_argument('--workers', type=int, default=6, help='The number of workers to use in a data loader (so training it bottelnecked by GPU not CPU')

        # For data augmentation
        self._parser.add_argument('--augment_training_data', default=True, type=bool, help='Shoudl data be augmented in training?')
        self._parser.add_argument('--add_random_masking', action='store_true', help='Option to turn of random masking as part of data augmentation (but keep the rest)')
        self._parser.add_argument('--mask_prob', default=0.5, help='The probability for which to add a mask with random masking')
        self._parser.add_argument('--orientation_prob', default=0.5, help='The probability a random mask is a vertical bar (rather than horizontal bar)')
        self._parser.add_argument('--mean_valued_prob', default=0.5, help='The probability for which the mask is mean valued (rather than random noise)')
        self._parser.add_argument('--max_cover_ratio', default=0.5, help='The maximum ratio that we allow a mask to cover of the bounding around the joint positions')
        self._parser.add_argument('--noise_std', default=0.2, help='The stddev of the noise to add, if the mask is gaussian noise')

        # What optimizer to use
        self._parser.add_argument('--use_amsprop', action='store_true', help='If we want to use AMSProp instead of RMSProp for training')

        # Debugging
        self._parser.add_argument('--use_train_mode_to_eval', action='store_true')
        self._parser.add_argument('--batch_norm_momentum', type=float, default=0.5)

        # ===============================================================
        #                     Hourglass data processing options
        # ===============================================================
        self._parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                            help='flip the input during validation')
        self._parser.add_argument('--sigma', type=float, default=1,
                            help='Groundtruth Gaussian sigma.')
        self._parser.add_argument('--sigma_decay', type=float, default=0,
                            help='Sigma decay rate for each epoch.')
        self._parser.add_argument('--label_type', metavar='LABELTYPE', default='Gaussian',
                            choices=['Gaussian', 'Cauchy'],
                            help='Labelmap dist type: (default=Gaussian)')
        self._parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')

        # ===============================================================
        #                     2D3D model options
        # ===============================================================
        self._parser.add_argument('--max_norm',       dest='max_norm', action='store_true', help='maxnorm constraint to weights')
        self._parser.add_argument('--linear_size',    type=int, default=1024, help='size of each model layer')
        self._parser.add_argument('--num_stage',      type=int, default=2, help='# layers in linear model')

        # ===============================================================
        #                     2D3D training options
        # ===============================================================
        self._parser.add_argument('--use_hg',         dest='use_hg', action='store_true', help='whether use 2d pose from hourglass')
        self._parser.add_argument('--job',            type=int,    default=8, help='# subprocesses to use for data loading')
        self._parser.add_argument('--no_max',         dest='max_norm', action='store_false', help='if use max_norm clip on grad')
        self._parser.add_argument('--max',            dest='max_norm', action='store_true', help='if use max_norm clip on grad')
        self._parser.set_defaults(max_norm=True)
        self._parser.add_argument('--procrustes',     dest='procrustes', action='store_true', help='use procrustes analysis at testing')

        # ===============================================================
        #                     2D3D training options (Cycle GAN/Data Augmentation)
        # ===============================================================
        self._parser.add_argument('--cycle_gan', action='store_true', help='if we would like to use a cycle gan for 2D->3D pose training')
        self._parser.add_argument('--discr_updates_per_gen_update', default=1, help='The number of updates to make for the discriminators per update for the generators')
        self._parser.add_argument('--gan_coeff', type=float, default=1.0, help='The coefficient of the gan')
        self._parser.add_argument('--cycle_coeff', type=float, default=10.0, help='The value of lambda in cycle gan, the weighting of the cycle consistency loss')
        self._parser.add_argument('--regression_coeff', type=float, default=10.0, help='Coeff for regression losses in cycle GAN, set to 0 if want to ignore. ')
        self._parser.add_argument('--use_fc_for_projection', action='store_true', help='If we want to replate the learnable projection by a ')

        self._parser.add_argument('--project_lr', type=float, default=-1.0, help='Option to override the learning rate for the projection net (cycle gan)')
        self._parser.add_argument('--discr_2d_lr', type=float, default=-1.0, help='Option to override the learning rate for the 2D pose discriminator (cycle gan)')
        self._parser.add_argument('--discr_3d_lr', type=float, default=-1.0, help='Option to override the learning rate for the 3D pose discriminator (cycle gan)')

        self._parser.add_argument('--orthogonal_data_augmentation', action='store_true', help="If we would like to perform the orthogonal pose augmentation.")
        self._parser.add_argument('--z_rotations_only', action='store_true', help='If the orthogonal data augmentation should only rotate about the z axis.')
        self._parser.add_argument('--dataset_normalization', action='store_true', help="If we want to revert to using dataset statistics for normalizing the input to the network, rather than normalizing per instance")
        self._parser.add_argument('--flip_prob', type=float, default=0.5, help="In the orthogonal data augmentation, the probability of performing a flip/reflection.")
        self._parser.add_argument('--drop_joint_prob', type=float, default=0.0, help="The probability of dropping each joint (independently) as input to the 3D baseline network.")

        # ===============================================================
        #                     "Generative models" training options
        # ===============================================================
        self._parser.add_argument('--gradient_penalty_coeff', type=float, default=10.0, help='Coeff for gradient penalty losses in a GAN, set to 0 if want to ignore.')
        self._parser.add_argument('--num_discriminator_steps_per_generator_step', type=int, default=5, help='the number of steps to update the discriminator for per generator update')
        self._parser.add_argument('--clip_max', type=float, default=0.01, help='The maximum magnitude of a parameter in the WGAN (used for parameter clipping).')

        # ===============================================================
        #                     "Stitched" training options
        # ===============================================================
        self._parser.add_argument('--load_hourglass',  type=str, help='Checkpoint file for a pre-trained hourglass model.')
        self._parser.add_argument('--load_2d3d',       type=str, help='Checkpoint file for a pre-trained 2d3d model')



    def _print(self):
        # Pretty prints the options we parses
        print("\n==================Options=================")
        pprint(vars(self._opt), indent=4)
        print("==========================================\n")



    def parse(self):
        # Parse the (known) arguments (with respect to the parser). Provide defaults appropriately
        training_key = self._script_id if self._script_id in training_defaults.keys() else ""
        viz_key = self._script_id if self._script_id in visualize_defaults.keys() else ""
        print("Using '%s' key for training default params." % training_key)
        print("Using '%s' key for visualizing default params." % viz_key)
        self._initial(training_defaults[training_key], visualize_defaults[viz_key])
        self._opt, _ = self._parser.parse_known_args()

        # Perform some validation on input
        checkpoint_dir = os.path.join(self._opt.checkpoint_dir, self._opt.exp)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if self._opt.load:
            if not os.path.isfile(self._opt.load):
                print ("{} is not found".format(self._opt.load))

        # Set internal variables
        self._opt.is_train = False if self._opt.test else True
        self._opt.checkpoint_dir = checkpoint_dir

        # PPrint options parsed (sanity check for user)
        self._print()
        return self._opt
