# Future/Compatability with Python2 and Python3
from __future__ import print_function, absolute_import, division

# Import matplotlib and set backend to agg so it doesn't go wrong
# MUST be first, before ANY file imports numpy for example
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Relative imports
from stacked_hourglass.evaluation.eval_PCKh import compute_PCKh_curve
from utils.osutils import mkdir_p, isdir

# Absolute imports
import sys
from options import Options
import os
import scipy
import numpy as np



def graph_PCKh_scores(options):
    """
    Script that takes a list of predictions and plots the PCKh curves and saves the figure to a file

    Required options:
    options.prediction_files - a space seperated list of prediction files (output as part of the model checkpointing)
    options.model_names - a space seperated list of model names, used in the figure
    options.output_dir - specifies a directory to save the graph as an image as

    :param options: Options for the evaluation, defined in options.py. (Including defaults).
    """
    pred_files = options.prediction_files
    model_names = options.model_names

    if len(pred_files) != len(model_names):
        raise Exception("Options must be the same length")

    if not isdir(options.output_dir):
        mkdir_p(options.output_dir)

    curves = {}
    for i in range(len(model_names)):
        curves[model_names[i]] = compute_PCKh_curve(pred_files[i], model_names[i])

    for key in curves[model_names[0]]:
        fig = plt.figure(figsize=(10.0, 10.0))
        for model in model_names:
            plt.plot(np.arange(0.0, 0.5, 0.01), curves[model][key], label=model)
        plt.legend()
        plt.xlabel("Threshold")
        plt.ylabel("% joints correct")
        plt.title("PCKh curves")

        # convert fig to a numpy array
        # see: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # avoid unecessary memory consumption
        plt.close(fig)

        # save
        filename = os.path.join(options.output_dir,"graph_{joint}_PCKh.jpg".format(joint=key))
        scipy.misc.imsave(filename, data)





if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "mpii_PCKh":
        graph_PCKh_scores(options)
    else:
        raise NotImplementedError()
