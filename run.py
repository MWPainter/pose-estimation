# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Relative imports
from stacked_hourglass import run as run_hourglass
from twod_threed import run as run_twod_to_threed

# Absolute imports
import sys
from options import Options
import os



def hourglass(options):
    """
    Script to run a trained hourglass model on an entire dataset.
    options.load: specifies the location of the saved model
    options.data_dir: specifies the location of the dataset to make 2D pose predictions on
    options.output_dir: specifies the location to save the (torch dataset of) pose predictions

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    run_hourglass(options)

def twod_to_threed(options):
    """
    Script to run a trained 2D to 3D pose model on an entire dataset.
    options.load: specifies the location of the saved model
    options.data_dir: specifies the location of the dataset to make 2D pose predictions on
    options.output_dir: specifies the location to save the (torch dataset of) pose predictions

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    run_twod_to_threed(options)



    :return: PyTorch Dataset object of 2D pose predictions
        # Get info about the file
if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "hourglass":
        hourglass(options)
    elif script == "2d3d":
        twod_to_threed(options)
    elif script == "stitched":
        raise NotImplementedError()
