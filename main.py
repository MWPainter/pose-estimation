from twod_threed import main as twod_threed_main
import sys
import os



def train_hourglass(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.

    Options:
    --id: the id of the network we are training (so we can train multiple vesions of the same network without clobbering)
    TODO: finish this list

    :param options: Options for the training, defined above.
    """
    raise NotImplementedError()



def train_twod_to_threed(options):
    """
    Script to create a stacked hourglass network, and train it from scratch to make 2D pose estimations.

    Options:
    --id: the id of the network we are training (so we can train multiple vesions of the same network without clobbering)
    TODO: finish this list

    :param options: Options for the training, defined above.
    """
    twod_threed_main(option)



if __name__ == "__main__":
    # Check that a script was specified
    if len(sys.argv) < 2:
        raise RuntimeException("Need to provide an argument specifing the 'script' to run.")

    # get args from command line
    script = sys.argv[1]
    options = Options(script).parse()

    # run the appropriate 'script'
    if script == "hourglass":
        # TODO: options object defaults
        train_hourglass(options)
    elif script == "2D3D":
        # TODO: options ovject defaults
        train_twod_to_threed(options)
