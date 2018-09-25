import sys
import torch

# human3.6m index to mpii index
# so if we want the ith joint (in human3.6m's indexing) from joints that are mpii indexed. Use joints[htm_idx[i]]
# to really understand this mapping, draw a skeleton out and label each joint with the two schemes
h36m_to_mpii_idx = [3, 2, 1, 4, 5, 6, 0, 7, 8, 9, 15, 14, 13, 10, 11, 12]


def mpii_to_h36m_joints_single(joints):
    """
    Transform a single set of joints according to 'mpii_to_h36m_joints'

    :param joints: Either a 16x2 2D vector (torch.Tensor) or a 32 1D vector, representing 16 MPII indexed joints
    :return: Either a 16x2 2D vector (torch.Tensor), representing 16 Human3.6m indexed joints
    """
    return mpii_to_h36m_joints(joints.unsqueeze_(0)).squeeze()



def mpii_to_h36m_joints(joints):
    """
    Transform points that are indexed in the mpii format, and convert them to the h36m format. This function
    works by using "integer array indexing" from https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html

    :param joints: Either a batch of 16x2 2D vector (torch.Tensor) or a 32 1D vector, representing 16 MPII indexed joints
    :return: Either a 16x2 2D vector (torch.Tensor), representing 16 Human3.6m indexed joints
    """
    reshape_req = joints.dim() == 2
    if reshape_req:
        joints = joints.view((-1,16,2))

    h36m_joints = joints[:, h36m_to_mpii_idx]

    if reshape_req:
        h36m_joints = h36m_joints.view((-1,32))

    return h36m_joints



def transform_video_mpii_to_h36m(joint_timeseries):
    """
    Transforms a timeseries (corresponding to a video) of joints from MPII indexing to

    :param joint_timeseries: Given a set of T joints (corresponding to a video), transform all of the indexing from
        mpii indexing to h36m indexing.
    :return: The same joint timeseries, but shuffled so that the points correspond to human3.6m indexing rather than
        MPII indexing.
    """
    return torch.stack([mpii_to_h36m_joints(joints) for joints in joint_timeseries])



def transform_dataset_mpii_to_h36m(dataset_file, output_file, is_video):
    """
    Transform an entire dataset of points.

    :param dataset_file: PyTorch file containing a dictionary of examples (MPII indexed)
    :param output_file: Where to save a PyTorch file containing the new dictionary indexed in h36m format
    :param is_video: If the examples are videos
    :return: Nothing. It saves a file
    """
    dataset = torch.load(dataset_file)

    new_dataset = {}
    for key in dataset:
        if is_video:
            new_dataset[key] = transform_video_mpii_to_h36m(dataset[key])
        else:
            new_dataset[key] = mpii_to_h36m_joints(dataset[key])

    torch.save(new_dataset, output_file)



if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Must take 3 arguments. Dataset input filename, dataset output filename and if data is in video format")

    dataset_file = sys.argv[1]
    output_file = sys.argv[2]
    is_video = bool(sys.argv[3])

    transform_dataset_mpii_to_h36m(dataset_file, output_file, is_video)