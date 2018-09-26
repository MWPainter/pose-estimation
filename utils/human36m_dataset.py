from __future__ import print_function, absolute_import

import copy
import math
import os
import pickle
import random
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset

import utils.camera_utils as camera_utils
import utils.data_utils as data_utils
from utils.osutils import *


DATASET_PATH = "/data/h36m_pose"
CAMERA_FILE = DATASET_PATH + "/cameras.h5"
ALL_ACTIONS = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

class Human36mDataset(Dataset):
    """
    A class containing all of the dataset logic for (image, 2D_out, 2D_normalized_in, 3D_out) tuples. Where the first
    two items are for training stacked hourglass networks, and the later two iterms are for training the 3D baseline
    network.

    Poses are stored in *world coordinates*, and cameras are stored for poses. Any other formats are computed on
    the fly when and as they are needed.

    Instance normalization refers to if the data will be normalized per instance, rather than using dataset statistics.

    Importantly, note that there are 17 moving joints in Human3.6m. For the 2D poses, we ignore the Neck/Nose joint.
    Therefore we have 16 joints at the 2D poses level, and 17 for the 3D poses.

    self.imgs, self.pose and self.pose_meta are arrays, of length equal to the number of examples in the dataset
    self.cams will be indexed by (subject_id, camera_id) and is a dictionary
    """

    def __init__(self, camera_file=CAMERA_FILE, dataset_path=DATASET_PATH, cams_per_frame=4, is_train=True,
                 orthogonal_data_augmentation=False, z_rotations_only=False, dataset_normalization=False, num_joints=32,
                 num_joints_pred_2d=16, num_joints_pred_3d=17, flip_prob=0.5, drop_joint_prob=0.0):
        # remember the parameters input
        self.camera_file = camera_file
        self.dataset_path = dataset_path
        self.cams_per_frame = cams_per_frame
        self.is_train = is_train
        self.orthogonal_data_augmentation = orthogonal_data_augmentation
        self.z_rotations_only = z_rotations_only
        self.dataset_normalization = dataset_normalization
        self.num_joints = num_joints
        self.num_joints_pred_2d = num_joints_pred_2d
        self.num_joints_pred_3d = num_joints_pred_3d
        self.flip_prob = flip_prob
        self.drop_joint_prob = drop_joint_prob

        # The Human3.6m subjects to use for training/validation
        self.subjects = data_utils.TRAIN_SUBJECTS if is_train else data_utils.TEST_SUBJECTS

        # Load in the images, 3D poses and cameras
        self.train_imgs, self.val_imgs = self._load_imgs(dataset_path)
        self.train_pose, self.train_pose_meta, self.val_pose, self.val_pose_meta = self._load_pose(dataset_path)
        self.train_cams = self._load_cams(camera_file, data_utils.TRAIN_SUBJECTS)
        self.val_cams = self._load_cams(camera_file, data_utils.TEST_SUBJECTS)

        self.imgs = self.train_imgs if is_train else self.val_imgs
        self.cams = self.train_cams if is_train else self.val_cams
        self.pose = self.train_pose if is_train else self.val_pose
        self.pose_meta = self.train_pose_meta if is_train else self.val_pose_meta

        # Dimensions to actually use from the Human3.6m data in the models
        self.pose_2d_indx_to_use, self.pose_2d_indx_to_ignore = data_utils.dimensions_to_use(is_2d=True)
        self.pose_3d_indx_to_use, self.pose_3d_indx_to_ignore = data_utils.dimensions_to_use(is_2d=False)

        # Compute normalization stats
        self.imgs_mean, self.imgs_std = self._compute_norm_stats_imgs()
        print("About to compute normalization stats")
        self.pose_3d_mean, self.pose_3d_std, self.pose_2d_mean, self.pose_2d_std = self._compute_norm_stats_poses()
        print("Computed normalization stats")

        # Only remember the means and std's of the dimensions we're actually going to use
        # And add a small constant to the std, to avoid div by zero errors
        self.pose_3d_mean = self.pose_3d_mean[self.pose_3d_indx_to_use]
        self.pose_3d_std  = self.pose_3d_std[self.pose_3d_indx_to_use] + 1.0e-8
        self.pose_2d_mean = self.pose_2d_mean[self.pose_2d_indx_to_use]
        self.pose_2d_std  = self.pose_2d_std[self.pose_2d_indx_to_use] + 1.0e-8




    def _print_camera_params(self):
        for param in range(len(self.cams[(self.subjects[0], 1)])):
            print("")
            if param == 0: print("Rotations (R):")
            elif param == 1: print("Translations (T):")
            elif param == 2: print("Focal lengths (f):")
            elif param == 3: print("Camera centers (c):")
            elif param == 4: print("Radial distortions (k):")
            elif param == 5: print("Tangental distortions (p):")
            elif param == 6: print("Names:")

            for key in self.cams:
                print(self.cams[key][param])



    def _load_imgs(self, img_path):
        # TODO: actually implement this
        return [], []



    def _load_image_from_filename(self, img_file):
        # TODO: actually implement this. (Necessary as all of the images are likely too large to store in memory).
        return None



    def _load_cams(self, camera_file, subjects):
        """
        Uses the camera_utils to load all of the camera objects.

        :param camera_file: The file 'cameras.h5', containing the camera parameters
        :param subjects: A list of subjects from Human3.6m that we want to load the cameras for
        :return: A dictionary, indexed by (subject number, cam number) keys
        """
        return camera_utils.load_cameras(camera_file, subjects)



    def _load_pose(self, dataset_path):
        """
        Uses the data_utils to load all of the 3D poses in Human3.6m, in 'world coordinates'. We also keep around the
        meta data for each of the poses, which are tuples (subject, action, sequence_id).

        We flatten all of the poses to be a 2D array with shape (total frames, 96).
        Meta data contains the subject number, action, sequence id and frame index (of the video) to recover all
        necessary information.

        :param dataset_path: The directory for which the dataset is stored.
        :return: train_poses, train_meta, val_poses, val_meta
        """
        # Load 3d data
        rtrain_set = data_utils.load_data(dataset_path, data_utils.TRAIN_SUBJECTS, ALL_ACTIONS, dim=3)
        rval_set = data_utils.load_data(dataset_path, data_utils.TEST_SUBJECTS, ALL_ACTIONS, dim=3)

        # Convert into friendly indexed training and test set
        train_set, train_set_meta = self._flatten_poses_set(rtrain_set)
        val_set, val_set_meta = self._flatten_poses_set(rval_set)

        return train_set, train_set_meta, val_set, val_set_meta



    def _flatten_poses_set(self, dict):
        """
        Given a dict, where each item has a key of format (subject, action, sequence_id), and value is a
        2D array (indexed by frame and then joint indices) corresponding to video data of poses.

        :param dict: A dictionary to convert.
        :return: list of poses, meta data about those poses
        """
        poses = []
        meta = []
        for key in dict:
            for frame_indx in range(len(dict[key])):
                poses.append(dict[key][frame_indx])
                subject_no, action, sequence_id = key
                meta_dict = {"subject_number": subject_no,
                             "action": action,
                             "sequence_id": sequence_id,
                             "frame_index": frame_indx}
                meta.append(meta_dict)
        return poses, meta



    def _flattened_train_cameras(self):
        """
        Return a 2D list, where the ith item is a list of cameras to use with self.pose[i].
        :return:
        """
        camss = []
        for frame in range(len(self.train_pose)):
            cams = []
            subject = self.train_pose_meta[frame]["subject_number"]
            for cam_indx in range(1,5):
                cams.append(self.train_cams[(subject, cam_indx)])
            camss.append(cams)
        return camss



    def _compute_norm_stats_imgs(self):
        # TODO: actually implment this
        # TODO: need to be able to actually override this, so that we can normalize in the same way MPII is
        return [0.0], [1.0]



    def _compute_norm_stats_poses(self):
        """
        Computes the mean and std dev of the poses in the *training* dataset. EVEN if this is a validation dataset.
        This just puts together a bunch of stuff from data_utils really.

        A very important part of this function, and why we *always* run it even if we are using instance normalization
        is to get the dimensions to use for the 2D and 3D poses. Note that there are 17 moving joints in Human3.6m.
        For the 2D poses, we ignore the Neck/Nose joint. Therefore we have 16 joints at the 2D poses level, and 17
        for the 3D poses.

        :return: (stats_2d, stats_3d), where each of stats_2d/stats_3d is a tuple with the following four things:
            mean = the mean of the training dataset
            std = the std dev of the training dataset
            dims_to_ignore = unused indices from the h36m data (some joints don't move)
            dims_to_use = used indices from the h36m data (joints indices that do move from the data)
        """
        # Load from cache if it exists
        cache_dir = ".cache/"
        cache_file = join(cache_dir, "h36m_pose_stats")
        if isfile(cache_file):
            print("loading h36m pose stats from cache file: " + cache_file)
            return pickle.load(open(cache_file, "rb"))

        # Transform all of the coords to camera space, and compute stats
        poses_camera_coords = self.world_to_camera_3d(self.train_pose, self._flattened_train_cameras())
        poses_transformed, _ = data_utils.postprocess_3d(poses_camera_coords)
        poses_transformed_np = copy.deepcopy(np.vstack(poses_transformed))
        mean_3d, std_3d = data_utils.normalization_stats(poses_transformed_np, dim=3)

        # Project all of the the points, and compute the stats for 2d coords
        poses_projected = self.project_poses(self.train_pose, self._flattened_train_cameras())
        poses_projected_np = copy.deepcopy(np.vstack(poses_projected))
        mean_2d, std_2d = data_utils.normalization_stats(poses_projected_np, dim=2)

        # Cache the stats (as they're slow to compute)
        print("finished computing h36m pose stats, caching to file: " + cache_file)
        if not isdir(cache_dir):
            mkdir_p(cache_dir)
        stats = (mean_3d, std_3d, mean_2d, std_2d)
        pickle.dump(stats, open(cache_file, "wb"))

        return mean_3d, std_3d, mean_2d, std_2d



    def world_to_camera_3d(self, poses, cams):
        """
        Convert poses in the 'world coordinates' into the 'camera coordinates'

        :param poses: An array of n poses in world coordinates
        :param cams: An array of size n (one per pose). Each item is another array of cameras (say, length m)
        :return: An array of size n*m of poses in camera coordinates
        """
        return data_utils.transform_world_to_camera(poses, cams)



    def world_to_camera_single_pose_3d(self, pose, cam):
        return self.world_to_camera_3d([pose], [[cam]])[0]



    def _rand_normals_3d(self, n):
        """
        Produce a batch of n random normal vectors, so the output will be of shape (n,3) with vectors on the unit sphere

        Implemented according to the following:
        https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices
        https://en.wikipedia.org/wiki/Spherical_coordinate_system

        :param n: The number of random vectors to compute
        :return: A Numpy tensor of random normals, with shape (n,3)
        """
        # Generate random numbers
        u_1 = np.random.uniform(size=n)
        u_2 = np.random.uniform(size=n)

        # Compute theta and phi for random spherical coordinates
        theta = np.arccos(2.0 * u_1 - 1.0)
        phi = 2 * math.pi * u_2

        # Compute random normals
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s_phi = np.sin(phi)
        c_phi = np.cos(phi)

        normals = np.zeros((n, 3))
        normals[:, 0] = s_theta * c_phi
        normals[:, 1] = s_theta * s_phi
        normals[:, 2] = c_theta

        return normals



    def rand_orthogonal_transform_matrix(self):
        """
        Compute a random orthogonal matrix. (Not uniformly random, that's overly complex).

        :return: A Numpy tensor of shape (3,3) representing a random rotation and flip.
        """
        # Compute a random rotation matrix, from a random normal and random number in the rand [0,2pi)
        angle = 2.0 * random.random() * math.pi
        normal = np.array([[0.0, 0.0, 1.0]]) if self.z_rotations_only else self._rand_normals_3d(1)
        Q = data_utils.rotation_matrices(normal, angle)[0]

        # Decide if we are flipping randomly (in the xaxis)
        pr = random.random()
        if pr < self.flip_prob:
            Q = np.matmul(Q, np.diag([-1,1,1]))

        return Q



    def apply_orthogonal_transform_3d(self, pose, Q):
        """
        Given a 3D pose 'pose', apply the orthongonal transform represented by 'Q' on it

        Q is applied around the hip joint, not the origin

        :param pose: Numpy tensor of shape (k*3,), where there are
        :param Q: Numpy tensor of shape (3,3), representing a linear (orthogonal) transform to apply
        :return: Returns Q applied to each pose. I.e. pose * Q.T, as poses are stored as row vectors
        """
        pose_centered, hip_positions = data_utils.zero_hip_joint(pose, self.num_joints)
        pose_centered = np.reshape(pose_centered, (-1,3))
        pose_transformed_centered = np.matmul(pose_centered, Q.T)
        pose_transformed = pose_transformed_centered + hip_positions
        return pose_transformed.flatten()



    def project_poses(self, poses, cams):
        """
        Project poses[i] onto all cameras in cams[i]. Returning a big array of all of the projections

        :param poses: An array of poses, in world coordinates (length n)
        :param cams: An array of camera parameters to use for the projections (length m)
        :return: An array of projected (2d) poses, in camera coordinates (length n*m)
        """
        return data_utils.project_to_cameras(poses, cams)


    def project_pose_3d(self, pose, cam):
        """
        Project 'pose' onto camera 'cam'

        :param pose: A single 3D pose, in world coordinates
        :param cam: The camera parameters to use for the projection
        :return: 2D pose coordinates, of 'pose' projected to camera 'cam'
        """
        return self.project_poses([pose], [[cam]])[0]



    def normalize_single_pose(self, pose_camera_coords, num_joints, is_2d):
        """
        Normalize a single pose (see data_utils function)
        """
        if is_2d:
            return data_utils.normalize_single_pose(pose_camera_coords, num_joints, self.dataset_normalization,
                                                    self.pose_2d_mean, self.pose_2d_std, is_2d=is_2d)
        else:
            return data_utils.normalize_single_pose(pose_camera_coords, num_joints, self.dataset_normalization,
                                                    self.pose_3d_mean, self.pose_3d_std, is_2d=is_2d)


    def __getitem__(self, index):
        """
        Get the 'index'th item from the dataset, a tuple (img, pos_in_img, 2d_pose, 3d_pose, meta).

        meta is a dict of metadata that could be useful. The following information is provided in meta:
        index = the index of the item in the dataset
        cam = the camera object used in the projection
        Q = the orthogonal transform that was applied to the pose before projection
        
        This function performs the following:
        1. Gets data from the appropriate storage
        2. (if self.orthogonal_data_augmentation) Performs an orthogonal transformation around the hip joint
        3. Projects the pose to compute the 2D pose
        4. Subsample pose data, to only consider the dimensions that we really want to use
        5. Normalizes the 2D and 3D poses (either instance normalization, or, according to dataset statistics)
        6. Perform joint dropping/masking
        
        :param index: What index we want to get from the dataset, in the range [0, len(dataset)]
        :return: Returns the tuple (img, pos_in_img, 2d_pose, 3d_pose, meta), which are as follows:
            img = the image that we want to predict a 3D pose from
            pose_in_img = the 2D pose, in image coordinates of 'img'
            2d_pose = a normalized 2D pose, input to the 3D prediction network
            3d_pose = the 3D pose that we wish to predict
            meta = a dictionary of information that could be useful (defined above).
        """
        # Get the indices into the imgs/cams/pose
        frame_number = index // self.cams_per_frame
        camera_number = (index % self.cams_per_frame) + 1

        # Step 1, index into arrays
        # Get the image, camera and pose (in camera coordinates)
        img = None # todo: self._load_image_from_filename(self.imgs[frame_number])
        subject = self.pose_meta[frame_number]["subject_number"]
        cam = self.cams[(subject,camera_number)]
        pose = self.pose[frame_number]
        pose_in_img = None # todo: work out how to do this
        
        # Step 2, apply the (random) orthogonal transform
        Q = np.eye(3) if not self.orthogonal_data_augmentation else self.rand_orthogonal_transform_matrix()
        augmented_pose = self.apply_orthogonal_transform_3d(pose, Q)

        # Step 3, project the pose (this transforms the pose into camera coords and then projects)
        augmented_pose_2d = self.project_pose_3d(augmented_pose, cam)

        # Step 4, sub sample the joints, so that we only give the network the ones that move
        augmented_pose_2d = augmented_pose_2d[self.pose_2d_indx_to_use]
        augmented_pose = augmented_pose[self.pose_3d_indx_to_use]

        # Step 5, normalize the 2D and 3D poses. (Note that we need to manually transform into camera coords, and
        # 'project_pose_3d' includes this transformation before projection)
        normalized_pose_2d, hip_pos_2d, scale_2d = self.normalize_single_pose(augmented_pose_2d, self.num_joints_pred_2d, is_2d=True)
        augmented_pose_cam = self.world_to_camera_single_pose_3d(augmented_pose, cam)
        normalized_pose, hip_pos, scale_3d = self.normalize_single_pose(augmented_pose_cam, self.num_joints_pred_3d, is_2d=False)

        # Step 6, randomly drop some joints (only on the input/2D pose)
        joint_mask = np.array(np.random.uniform(size=self.num_joints) > self.drop_joint_prob, dtype=int)
        if self.drop_joint_prob > 0.0:
            normalized_pose_2d = np.reshape(normalized_pose_2d, (self.num_joints, -1))
            normalized_pose_2d *= np.expand_dims(joint_mask, axis=1)
            normalized_pose_2d = normalized_pose_2d.flatten()

        # Store any meta data
        meta = {
            'index': index,
            'frame_number': frame_number,
            'cam_number': camera_number,
            'cam': cam,
            'Q': Q,
            'joint_mask': joint_mask,
            '3d_pose_camera_coords': augmented_pose_cam,
            '2d_indx_used': self.pose_2d_indx_to_use,
            '3d_indx_used': self.pose_3d_indx_to_use,
            '2d_indx_ignored': self.pose_2d_indx_to_ignore,
            '3d_indx_ignored': self.pose_3d_indx_to_ignore,
        }
        if self.dataset_normalization:
            # to "unNormalize" in datasrt normalization
            meta.update({
                '2d_mean': self.pose_2d_mean,
                '3d_mean': self.pose_3d_mean,
                '2d_std': self.pose_2d_std,
                '3d_std': self.pose_3d_std,
            })
        else:
            # to "unNormalize" in instance normalization
            meta.update({
                '2d_hip_pos': hip_pos_2d,
                '3d_hip_pos': hip_pos,
                '2d_scale': scale_2d,
                '3d_scale': scale_3d,
            })

        # Return the tuple
        return (img, pose_in_img, torch.Tensor(normalized_pose_2d), torch.Tensor(normalized_pose), meta)



    def __len__(self):
        """ 
        There are 4 cameras for each pose
        """
        return len(self.pose) * 4.0


