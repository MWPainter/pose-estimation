# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Relative imports
from generative_models import train_3d_pose_gan as threed_pose_gan_main
from twod_threed import main as twod_threed_h36m_main
from stacked_hourglass import mpii_main as hourglass_mpii_main

# Absolute imports
import sys
from options import Options
import os
import random
import numpy as np
import torch



def train_hourglass_mpii(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.
    Options were taken mostly from from the "twod_threed" code. We've merged the options to make it easier.
    We add a couple arguments here, without specifying the option to change them, as in this context it doesn't
    make sense to allow them to be changed.

    The default options for the hourglass code can be found at the bottom of ./stacked_hourglass/example/mpii.py.

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    options.arch = "hg"
    hourglass_mpii_main(options)



def train_twod_to_threed_h36m(options):
    """
    Script to create a "3D pose baseline" network, and train it..

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    twod_threed_h36m_main(options)



def train_3d_pose_gan(options):
    """
    Script to train a gan for 3d poses with human3.6m

    Required args.
    options.data_dir: specifies the location of the dataset to make 2D pose predictions on
    options.exp: experiment id

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    threed_pose_gan_main(options)



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
    if script == "hourglass_mpii":
        train_hourglass_mpii(options)
    elif script == "2d3d_h36m":
        train_twod_to_threed_h36m(options)
    elif script == "3d_pose_gan":
        train_3d_pose_gan(options)
    elif script == "stitched":
        raise NotImplementedError()
    else:
        raise NotImplementedError()
