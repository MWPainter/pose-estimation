"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import camera_utils as cameras
import h5py
import glob
import copy
import torch

"""
FILE ORIGINALLY PART OF THE 3D BASELINE CODE (twod_threed library)
"""

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'           # 0
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

adj = [
    (0,1), #  hip -> rhip
    (1,2), # rhip -> rkne
    (2,3), # rkne -> rank
    ()
]

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = [''] * 16
SH_NAMES[0] = 'RFoot'
SH_NAMES[1] = 'RKnee'
SH_NAMES[2] = 'RHip'
SH_NAMES[3] = 'LHip'
SH_NAMES[4] = 'LKnee'
SH_NAMES[5] = 'LFoot'
SH_NAMES[6] = 'Hip'
SH_NAMES[7] = 'Spine'
SH_NAMES[8] = 'Thorax'
SH_NAMES[9] = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'


def load_data(bpath, subjects, actions, dim=3):
    """
    Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

    Args
      bpath: String. Path where to load the data from
      subjects: List of integers. Subjects whose data will be loaded
      actions: List of strings. The actions to load
      dim: Integer={2,3}. Load 2 or 3-dimensional data
    Returns:
      data: Dictionary with keys k=(subject, action, seqname)
        values v=(nx(32*2) matrix of 2d ground truth)
        There will be 2 entries per subject/action if loading 3d data
        There will be 8 entries per subject/action if loading 2d data
    """

    if not dim in [2, 3]:
        raise (ValueError, 'dim must be 2 or 3')

    data = {}

    for subj in subjects:
        for action in actions:
            print('Reading subject {0}, action {1}'.format(subj, action))

            dpath = os.path.join(bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim),
                                 '{0}*.h5'.format(action))
            print(dpath)

            fnames = glob.glob(dpath)

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename(fname)

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith(action):
                    print(fname)
                    loaded_seqs = loaded_seqs + 1

                    with h5py.File(fname, 'r') as h5f:
                        poses = h5f['{0}D_positions'.format(dim)][:]

                    poses = poses.T
                    data[(subj, action, seqname)] = poses

            if dim == 2:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format(loaded_seqs)
            else:
                assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format(loaded_seqs)

    return data


def load_stacked_hourglass(data_dir, subjects, actions):
    """
    Load 2d detections from disk, and put it in an easy-to-acess dictionary.

    Args
      data_dir: string. Directory where to load the data from,
      subjects: list of integers. Subjects whose data will be loaded.
      actions: list of strings. The actions to load.
    Returns
      data: dictionary with keys k=(subject, action, seqname)
            values v=(nx(32*2) matrix of 2d stacked hourglass detections)
            There will be 2 entries per subject/action if loading 3d data
            There will be 8 entries per subject/action if loading 2d data
    """
    # Permutation that goes from SH detections to H36M ordering.
    SH_TO_GT_PERM = np.array([SH_NAMES.index(h) for h in H36M_NAMES if h != '' and h in SH_NAMES])
    assert np.all(SH_TO_GT_PERM == np.array([6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10]))

    data = {}

    for subj in subjects:
        for action in actions:

            print('Reading subject {0}, action {1}'.format(subj, action))

            dpath = os.path.join(data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action))
            print(dpath)

            fnames = glob.glob(dpath)

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename(fname)
                seqname = seqname.replace('_', ' ')

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith(action):
                    print(fname)
                    loaded_seqs = loaded_seqs + 1

                    # Load the poses from the .h5 file
                    with h5py.File(fname, 'r') as h5f:
                        poses = h5f['poses'][:]

                        # Permute the loaded data to make it compatible with H36M
                        poses = poses[:, SH_TO_GT_PERM, :]

                        # Reshape into n x (32*2) matrix
                        poses = np.reshape(poses, [poses.shape[0], -1])
                        poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])

                        dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
                        dim_to_use_y = dim_to_use_x + 1

                        dim_to_use = np.zeros(len(SH_NAMES) * 2, dtype=np.int32)
                        dim_to_use[0::2] = dim_to_use_x
                        dim_to_use[1::2] = dim_to_use_y
                        poses_final[:, dim_to_use] = poses
                        seqname = seqname + '-sh'
                        data[(subj, action, seqname)] = poses_final

            # Make sure we loaded 8 sequences
            if (subj == 11 and action == 'Directions'):  # <-- this video is damaged
                assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj,
                                                                                                      action)
            else:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj,
                                                                                                      action)

    return data


def normalization_stats(complete_data, dim):
    """
    Computes normalization statistics: mean and stdev, dimensions used and ignored

    Args
      complete_data: nxd np array with poses
      dim. integer={2,3} dimensionality of the data
    Returns
      data_mean: np vector with the mean of the data
      data_std: np vector with the standard deviation of the data
    """
    if not dim in [2, 3]:
        raise (ValueError, 'dim must be 2 or 3')

    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    return data_mean, data_std


def dimensions_to_use(is_2d):
    """
    Computes the dimensions to use out of all of the H36m data, as not all of them move independently.

    Importantly, note that there are 17 moving joints in Human3.6m. For the 2D poses, we ignore the Neck/Nose joint.
    Therefore we have 16 joints at the 2D poses level, and 17 for the 3D poses.

    :param is_2d: If we want the dimensions to use for 2d data
    :return: dimensions to use (in the model), dimensions to ignroe (in the model)
    """
    if is_2d:
        dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dimensions_to_use)
    else:  # dim == 3
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                               dimensions_to_use * 3 + 1,
                                               dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3), dimensions_to_use)
    return dimensions_to_use, dimensions_to_ignore


def transform_world_to_camera(poses_3d, cams):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_3d: list with 3d poses
      cams: list of lists with cameras (cams[i] is the list of camers for poses_3d[i])
    Return:
      camera_poses_set: list with 3d poses in camera coordinate (of length equal to flattened cams array)
    """
    poses_cam_coords = []
    for i in range(len(poses_3d)):
        if (i+2) % 1000 == 0:
            print(str(i) + "/" + str(len(poses_3d)))
        pose_world_coords = poses_3d[i]
        for j in range(len(cams[i])):
            R, T, f, c, k, p, name = cams[i][j]
            pose_cam_coords = cameras.world_to_camera_frame(np.reshape(pose_world_coords, [-1, 3]), R, T)
            pose_cam_coords = np.reshape(pose_cam_coords, [-1])
            pose_cam_coords = np.reshape(pose_cam_coords, [-1])
            poses_cam_coords.append(pose_cam_coords)
    return poses_cam_coords


def normalize_data(data, data_mean, data_std):
    """
    Normalizes a list of poses

    Args
      data: list where values are
      data_mean: np vector with the mean of the data
      data_std: np vector with the standard deviation of the data
      dim_to_use: list of dimensions to keep in the data
    Returns
      data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = []
    for indx in range(len(data)):
        data_out.append(normalize_pose(data[indx], data_mean, data_std))
    return data_out


def normalize_pose(pose, data_mean, data_std):
    # data = pose[:, dim_to_use]
    # mu = data_mean[dim_to_use]
    # stddev = data_std[dim_to_use]
    return normalize(pose, data_mean, data_std)


def normalize(data, data_mean, data_std):
    """
    Normalize some data

    data shape = (*, K)
    data_mean shape = (K,)
    data_std shape = (K,)
    so that broadcasting happens correctly
    """
    return np.divide((data - data_mean), data_std)


def define_actions(action):
    """
    Given an action string, returns a list of corresponding actions.

    Args
      action: String. either "all" or one of the h36m actions
    Returns
      actions: List of strings. Actions to use.
    Raises
      ValueError: if the action is not a valid action in Human 3.6M
    """
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def project_to_cameras(poses_3d, cams):
    """
    Project 3d poses using camera parameters

    Args
      poses_3d: array of poses, shape of (num_poses, 3*num_joints)
      cams: list of lists with cameras (cams[i] is the list of camers for poses_3d[i])
    Returns
      t2d: dictionary with 2d poses (If cams has "shape" (m), then this returns a list of length (num_poses*m) 2d poses)
    """
    poses_2d = []

    for i in range(len(poses_3d)):
        if (i+2) % 1000 == 0:
            print(str(i) + "/" + str(len(poses_3d)))
        pose_3d = poses_3d[i]
        for j in range(len(cams[i])):
            R, T, f, c, k, p, name = cams[i][j]
            pose_2d, _, _, _, _ = cameras.project_point_radial(np.reshape(pose_3d, [-1, 3]), R, T, f, c, k, p)
            pose_2d = np.reshape(pose_2d, [-1])
            poses_2d.append(pose_2d)

    return poses_2d


def read_2d_predictions(actions, data_dir):
    """
    Loads 2d data from precomputed Stacked Hourglass detections

    Args
      actions: list of strings. Actions to load
      data_dir: string. Directory where the data can be loaded from
    Returns
      train_set: dictionary with loaded 2d stacked hourglass detections for training
      test_set: dictionary with loaded 2d stacked hourglass detections for testing
      data_mean: vector with the mean of the 2d training data
      data_std: vector with the standard deviation of the 2d training data
      dim_to_ignore: list with the dimensions to not predict
      dim_to_use: list with the dimensions to predict
    """

    train_set = load_stacked_hourglass(data_dir, TRAIN_SUBJECTS, actions)
    test_set = load_stacked_hourglass(data_dir, TEST_SUBJECTS, actions)

    complete_train = copy.deepcopy(np.vstack(train_set.values()))
    data_mean, data_std = normalization_stats(complete_train, dim=2)

    train_set = normalize_data(train_set, data_mean, data_std)
    test_set = normalize_data(test_set, data_mean, data_std)

    return train_set, test_set, data_mean, data_std


def create_2d_data(actions, data_dir, rcams):
    """
    Creates 2d poses by projecting 3d poses with the corresponding camera
    parameters. Also normalizes the 2d poses

    Args
      actions: list of strings. Actions to load
      data_dir: string. Directory where the data can be loaded from
      rcams: dictionary with camera parameters
    Returns
      train_set: dictionary with projected 2d poses for training
      test_set: dictionary with projected 2d poses for testing
      data_mean: vector with the mean of the 2d training data
      data_std: vector with the standard deviation of the 2d training data
      dim_to_ignore: list with the dimensions to not predict
      dim_to_use: list with the dimensions to predict
    """

    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    train_set = project_to_cameras(train_set, rcams)
    test_set = project_to_cameras(test_set, rcams)

    # Compute normalization statistics.
    complete_train = copy.deepcopy(np.vstack(train_set.values()))
    data_mean, data_std = normalization_stats(complete_train, dim=2)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std)
    test_set = normalize_data(test_set, data_mean, data_std)

    return train_set, test_set, data_mean, data_std


def read_3d_data(actions, data_dir, camera_frame, rcams, predict_14=False):
    """
    Loads 3d poses, zero-centres and normalizes them

    Args
      actions: list of strings. Actions to load
      data_dir: string. Directory where the data can be loaded from
      camera_frame: boolean. Whether to convert the data to camera coordinates
      rcams: dictionary with camera parameters
      predict_14: boolean. Whether to predict only 14 joints
    Returns
      train_set: dictionary with loaded 3d poses for training
      test_set: dictionary with loaded 3d poses for testing
      data_mean: vector with the mean of the 3d training data
      data_std: vector with the standard deviation of the 3d training data
      dim_to_ignore: list with the dimensions to not predict
      dim_to_use: list with the dimensions to predict
      train_root_positions: dictionary with the 3d positions of the root in train
      test_root_positions: dictionary with the 3d positions of the root in test
    """
    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    if camera_frame:
        train_set = transform_world_to_camera(train_set, rcams)
        test_set = transform_world_to_camera(test_set, rcams)

    # Apply 3d post-processing (centering around root)
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)

    # Compute normalization statistics
    complete_train = copy.deepcopy(np.vstack(train_set.values()))
    data_mean, data_std = normalization_stats(complete_train, dim=3, predict_14=predict_14)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std)
    test_set = normalize_data(test_set, data_mean, data_std)

    return train_set, test_set, data_mean, data_std, train_root_positions, test_root_positions


def postprocess_3d(poses_set):
    """
    Center 3d points around root

    Args
      poses_set: list with 3d data
    Returns
      poses_set: list with 3d data centred around root (center hip) joint
      root_positions: list with the original 3d position of each pose
    """
    root_positions = []
    for k in range(len(poses_set)):
        # Keep track of the global position
        root_positions.append(copy.deepcopy(poses_set[k][:3]))

        # Remove the root from the 3d position
        poses = np.reshape(poses_set[k], [-1,3])
        poses = poses - poses[0]
        poses_set[k] = np.reshape(poses, [-1])

    return poses_set, root_positions



def zero_hip_joints(poses, num_joints):
    """
    Given a set of poses, with shape (n, k*d), where n is the batch size, k is the number of joints and d is the
    dimension of the joints. It reshapes the poses to (n, k, d) and subtracts poses[:,0] from each poses[0:i].

    :param poses: A Numpy tensor of shape (n, k*d), representing a batch of n poses with k joints.
    :param num_joints: The number of joints k, that we are considering
    :return: (zeroed_poses, root_positions)
        zeroed_poses: the poses with the first joint subtracted from all of the joints (zero the first joint)
                      of shape (n, k*d)
        root_positions: the joints that were subtracted from each of the poses
                        of shape (n, d)
    """
    # Reshape
    batch_size = poses.shape[0]
    poses = np.reshape(poses, (batch_size, num_joints, -1))

    # Subtract + remember the root positions
    root_potisions = poses[:, 0]
    poses = poses - root_potisions

    # reshape and return
    return np.reshape(poses, (batch_size, -1)), root_potisions


def zero_hip_joint(pose, num_joints):
    """
    Single instance version of zero_hip_joints
    """
    zeroed_pose_batch, zeroed_root_pos_batch = zero_hip_joints(np.expand_dims(pose, 0), num_joints)
    return np.squeeze(zeroed_pose_batch), np.squeeze(zeroed_root_pos_batch)


def zero_hip_joints_torch(poses, num_joints):
    """
    Torch version of zero_hip_joints
    """
    batch_size = poses.size(0)
    poses = poses.view(batch_size, num_joints, -1)

    root_positions = poses[:, 0]
    poses = poses - root_positions

    return poses.view(batch_size, -1), root_positions


def std_distance(pose, num_joints):
    """
    Given a set of poses, with shape (n, k*d), where n is the batch size, k is the number of joints and d is the
    dimension of the joints. It reshapes the poses to (n, k), and computes the std dev of the joints distances
    to the origin. That is, it computes the stddev of ||poses[:,i]||_2 over i.

    :param pose: A Numpy tensor of shape (k*d,), representing a pose with k joints.
    :param num_joints: The number of joints in the pose
    :return: A Numpy tensor of shape (1,) representing the std dev of each poses joint distances
    """
    # Reshape
    pose = np.reshape(pose, (num_joints, -1))

    # Compute the L2 norms
    norms = np.sqrt(np.sum(pose ** 2, axis=1))
    return np.std(norms)



def std_distance_torch(poses, num_joints):
    """
    Same as "std_distance", but implemented in torch, and for a batch
    """
    batch_size = poses.size(0)
    poses = poses.view(batch_size, num_joints, -1)
    norms = torch.sqrt(torch.sum(poses ** 2, 2))
    return torch.std(norms, 1)



def std_distance_torch_3d(poses):
    """
    Same as "std_distance", but implemented in torch, for 3D poses only, and for a batch
    """
    batch_size = poses.size(0)
    poses = poses.view(batch_size, -1, 3)
    norms = torch.sqrt(torch.sum(poses ** 2, 2))
    return torch.std(norms, 1)



def _outer_product(u, v):
    """
    (Batch) compute the outer product of u and v

    :param u: A Numpy tensor of shape (n,3)
    :param v: A Numpy tensor of shape (n,3)
    :return: The outer product of u, v with shape (n, 3, 3)
    """
    n = u.shape[0]
    u_exp = np.reshape(u, (n,3,1))
    v_exp = np.reshape(v, (n,1,3))
    return np.matmul(u_exp,v_exp)


def _cross_matrix(u):
    """
    Produce [u]_x from u, the matrix that has the same effect as taking the cross product with u.
    See: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    :param u: Numpy tensor of shape (n,3) that we wish to compute the cross product matrix for
    :return: Numpy tensor of shape (n,3,3), the cross product matrices for each vector
    """
    n = u.shape[0]
    cross = np.zeros((n,3,3))
    cross[:,0,1] = -u[:,2]
    cross[:,0,2] =  u[:,1]
    cross[:,1,0] =  u[:,2]
    cross[:,1,2] = -u[:,0]
    cross[:,2,0] = -u[:,1]
    cross[:,2,1] =  u[:,0]
    return cross


def rotation_matrices(normals, angles):
    """
    Given a batch of n random normal vectors, and n angles, produce n rotation matrices, of an angle angles[i] around
    normal normals[i].

    Implemented according to the following:
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    :param normals: A Numpy tensor of shape (n,3) of normals for the rotation matrices
    :param angles: A Numpy tensor of shape (n,) of angles in the range [0,2pi) for the angles to rotate
    :return: A Numpy tesnor of shape (n,3,3) of
    """
    # Compute the component parts
    batch_size = normals.shape[0]
    eye = np.zeros((batch_size,3,3)) + np.eye(3)
    normals_cross = _cross_matrix(normals)
    normals_outer = _outer_product(normals, normals)
    sin = np.sin(angles)
    cos = np.cos(angles)

    # Compute the matrices
    return cos * eye + sin * normals_cross + (1.0-cos) * normals_outer


def reflection_matrices(normals):
    """
    Compute matrices that reflect points about plane with normals 'normals'. Point a, can be reflected about the
    plane with normal b (passing through the origin) using the formula a - 2 (b dot a) b = (I - 2bb.T)a.
    Therfor the reflection matrix is I - 2bb.T

    :param normals: Numpy tensor with shape (n,3) of plane normals
    :return: Numpy tensor of shape (n,3,3) representing matrices for the
    """
    return -2.0 * _outer_product(normals, normals) + np.eye(3)



def unNormalizeData(pose, meta, dataset_normalization, is_2d=False):
    """
    Unnormalize data, for both dataset and instance normalization schemes.

    :param pose: The 3D poses output from the 3D baseline network
    :param meta: Meta data for this batch
    :param dataset_normalization: If we are using dataset normalization (as opposed to instance normalization)
    :param is_2d: If we are unnormalizing 2d data
    :return: UnNormalized poses, in their camera coordinates
    """
    if dataset_normalization:
        if not is_2d:
            mean = meta['3d_mean']
            std = meta['3d_std']
            return pose * std + mean
        else:
            mean = meta['2d_mean']
            std = meta['2d_std']
            return pose * std + mean
    else:
        if not is_2d:
            batch_size = pose.shape[0]
            pose_coords = np.reshape(pose, (batch_size, -1, 3))
            scales = np.reshape(meta['3d_scale'], (batch_size,1,1)) # (batch_size,) -> (batch_size,1,1) to broadcast correctly
            hip_poss = np.reshape(meta['3d_hip_pos'], (batch_size,1,3)) # (batch_size, 3) -> (batch_size,1,3) to broadcast correctly
            return np.reshape(pose_coords * scales + hip_poss, (batch_size, -1))
        else:
            batch_size = pose.shape[0]
            pose_coords = np.reshape(pose, (batch_size, -1, 2))
            scales = np.reshape(meta['2d_scale'], (batch_size,1,1)) # (batch_size,) -> (batch_size,1,1) to broadcast correctly
            hip_poss = np.reshape(meta['2d_hip_pos'], (batch_size,1,2)) # (batch_size, 2) -> (batch_size,1,2) to broadcast correctly
            return np.reshape(pose_coords * scales + hip_poss, (batch_size, -1))



def normalize_single_pose(pose_camera_coords, num_joints, dataset_normalization, pose_mean=None, pose_std=None, is_2d=False):
    """
    Normalize a single pose to center the points around the hip joint, and then also scale the points (by a
    scalar) to make the distance std dev equal to 1. If not using instance normalization we just return the
    data normalized as it would have been before.

    Dataset normalization scheme falls back onto the 'normalize data' function. Note that in the old code, 3D poses
    were centered around a zeroed hip joint, but, 2D poses were not.

    :param pose_camera_coords: A Numpy tensor of shape (k*d,) where there are
    :param num_joints: The number of joints in the pose (i.e. the value d).
    :param dataset_normalization: If we are normalizing poses according to dataset statistics
    :param is_2d: Boolean saying if the data is 2d or 3d points
    :param pose_mean: Mean for poses (2d or 3d) in dataset
    :param pose_std: Std for poses (2d of 3d) in dataset
    :return: if using instance normalization then (pose_coords_hip_zeroed, hip_root_positions, joint_dist_std)
        pose_coords_hip_zeroed = pose coordinates, which have been translated to make the hip at origin and scaled
                so that the distances of the joints to the origin have std = 1. Shape (k*d,)
        hip_root_position = the original positions of the hip joint. Shape (d,)
        joint_dist_std = the scale which we *divided* by to make the std = 1. Shape (1,)

    """
    if not dataset_normalization:
        pose_zeroed_hip, hip_root_position = zero_hip_joint(pose_camera_coords, num_joints)
        joint_dist_std = std_distance(pose_zeroed_hip, num_joints)
        return pose_zeroed_hip / joint_dist_std, hip_root_position, joint_dist_std
    else:
        poses = [pose_camera_coords]
        if not is_2d:
            poses, _ = postprocess_3d(poses)
        return normalize_data(poses, pose_mean, pose_std)[0], None, None



def normalize_poses(poses, num_joints, dataset_normalization, pose_mean=None, pose_std=None, is_2d=False):
    """
    Batch version of normalize_single_pose, implemented in torch
    """
    if not dataset_normalization:
        poses_zeroed_hip, hip_root_positions = zero_hip_joints(poses, num_joints)
        joint_dist_stds = std_distance_torch(poses_zeroed_hip, num_joints)
        return poses_zeroed_hip / joint_dist_stds.view(-1,1), hip_root_positions, joint_dist_stds

    else:
        if not is_2d:
            poses, _ = zero_hip_joints_torch(poses)
        return (poses - pose_mean) / pose_std, None, None


def compute_3d_pose_error_distances(outputs, tars, meta, dataset_normalization=False, procrustes=False):
    """
    Given PyTorch variables, outputs and tars, the outputs and targets for the 3D baseline network respectively.
    Compute the distances between all of them, in the unormalized space

    :param outputs: Output variable from the network
    :param tars: Target variable for the network
    :param meta: Meta data, containing the statistics information required to "unnormalize"
    :param dataset_normalization: If we are using istance or dataset statistics to normalize
    :param procrustes: If we allow for a procrustes transform in the error analysis
    :return: A PyTorch tensor of shape (batch_size, num_joints) of distances between the outputs and targets
    """
    # calculate erruracy
    targets_unnorm = data_utils.unNormalizeData(tars.data.cpu().numpy(), meta, dataset_normalization)
    outputs_unnorm = data_utils.unNormalizeData(outputs.data.cpu().numpy(), meta, dataset_normalization)

    # Meta contains PyTorch tensors, so targets_unnorm and outputs_unnorm are PyTorch tensors
    targets_use = targets_unnorm.numpy()
    outputs_use = outputs_unnorm.numpy()

    if procrustes:
        for ba in range(inps.size(0)):
            gt = targets_use[ba].reshape(-1, 3)
            out = outputs_use[ba].reshape(-1, 3)
            _, Z, T, b, c = get_transformation(gt, out, True)
            out = (b * out.dot(T)) + c
            outputs_use[ba, :] = out.reshape(1, 51)

    sqerr = (outputs_use - targets_use) ** 2
    sqerr = np.reshape(sqerr, (sqerr.shape[0], 17, 3))
    distance = np.sqrt(np.sum(sqerr, axis=2))
    return distance