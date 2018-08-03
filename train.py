# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Relative imports
from twod_threed import main as twod_threed_main
from stacked_hourglass import mpii_main as hourglass_main

# Absolute imports
import sys
from options import Options
import os



def train_hourglass(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.
    Options were taken mostly from from the "twod_threed" code. We've merged the options to make it easier.
    We add a couple arguments here, without specifying the option to change them, as in this context it doesn't
    make sense to allow them to be changed.

    The default options for the hourglass code can be found at the bottom of ./stacked_hourglass/example/mpii.py.

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    options.arch = "hg"
    hourglass_main(options)



def train_twod_to_threed(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    twod_threed_main(options)



if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "hourglass":
        train_hourglass(options)
    elif script == "2d3d":
        train_twod_to_threed(options)
    elif script == "stitched":
        raise NotImplementedError()
