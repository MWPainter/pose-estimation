# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Relative imports
from stacked_hourglass import run as run_hourglass_mpii
from stitched import run as run_stitched_mpii
from twod_threed import run as run_twod_to_threed_h36m

# Absolute imports
import sys
from options import Options
import os



def hourglass_mpii(options):
    """
    Script to run a trained hourglass model on an entire dataset.
    options.load: specifies the location of the saved model
    options.data_dir: specifies the location of the dataset to make 2D pose predictions on
    options.output_dir: specifies the location to save the (torch dataset of) pose predictions

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    run_hourglass_mpii(options)



def twod_to_threed_h36m(options):
    """
    Script to run a trained 2D to 3D pose model on an entire dataset.
    options.load: specifies the location of the saved model
    options.data_dir: specifies the location of the dataset to make 2D pose predictions on
    options.output_dir: specifies the location to save the (torch dataset of) pose predictions

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    run_twod_to_threed_h36m(options)



def stitched_mpii(options):
    """
    Script to run trained models for stacked hourglass and 3D baseline, stitched together using a soft argmax
    options.load_hourglass: specifies a file containing the saved stacked hourglass model
    options.load_2d3d: specifies a file containing the saved 3D baseline model
    options.output_dir: specifies a location to output the 2D and 3D predictions made by the network

    :param options: Options for the training, defined in options.py. (Including defaults).
    """
    run_stitched_mpii(options)






if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "hourglass_mpii":
        hourglass_mpii(options)
    elif script == "2d3d_h36m":
        twod_to_threed_h36m(options)
    elif script == "stitched_mpii":
        stitched_mpii(options)
