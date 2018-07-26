import sys
import os



def train_hourglass(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.

    Options:
    --id: the id of the network we are training (so we can train multiple vesions of the same network without clobbering)

    :param options: Options for the training, defined above.
    """




if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # run the appropriate 'script'
    script = sys.argv[1]
    options = None  # TODO:

    if script == "hourglass":
        # TODO: options object defaults
        train_hourglass(options)
    else if script == "2D3D":
        # TODO: options ovject defaults
        train_twod_to_threed(options)
