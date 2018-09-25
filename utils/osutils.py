from __future__ import absolute_import

import os
import errno

def mkdir_p(dir_path):
    """
    Makedirs
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isfile(fname):
    """
    Check if a filename is a file
    """
    return os.path.isfile(fname) 


def isdir(dirname):
    """
    Check if a filename is a directory
    """
    return os.path.isdir(dirname)


def join(path, *paths):
    """
    Join filepaths
    """
    return os.path.join(path, *paths)
