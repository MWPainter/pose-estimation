import torch
import utils.data_utils as data_utils
from .soft_argmax import SoftArgmax2D
from stacked_hourglass.pose.models.hourglass import HourglassNet
from stacked_hourglass.pose.utils.evaluation import final_preds_post_processing
from twod_threed.src.model import LinearModel as Transform2D3DNet


def _identity(x):
    return x

class StitchedNetwork(torch.nn.Module):
    def __init__(self, hg_stacks, hg_blocks, hg_num_classes, hg_batch_norm_momentum, hg_use_layer_norm, hg_mean, hg_std,
                 width, height, linear_size=1024, num_stage=2, p_dropout=0.5, input_size=32, output_size=51,
                 transformer_fn=_identity, dataset_normlization=False):
        """
        Initialize the stitched network

        :param hg_*: Parameters for the stacked hourglass model
        :param hg_mean: The mean used for normalization of data for hourglass
        :param hg_std: The standard deviation used in normalization of data for hourglass
        :param width: The width of the input images
        :param height: The height of the input images
        :param <not transformer>: Parameters for the 3D baseline model
        :param transformer: A function to transform the joints from one format to another, defaults to an identity function
        :param dataset_normlization: If we should use dataset normalization, (rather than instance normalization) for input to the 3d baseline model
        """
        super(StitchedNetwork, self).__init__()

        self.hg_stacks = hg_stacks
        self.hg_blocks = hg_blocks
        self.hg_num_classes = hg_num_classes
        self.hg_batch_norm_momentum = hg_batch_norm_momentum
        self.hg_use_layer_norm = hg_use_layer_norm
        self.hg_mean = hg_mean
        self.hg_std = hg_std

        self.baseline_linear_size = linear_size
        self.baseline_num_stage = num_stage
        self.baseline_p_dropout = p_dropout
        self.baseline_input_size = input_size
        self.baseline_output_size = output_size

        self.baseline_dataset_normalization = dataset_normlization

        self.stacked_hourglass = HourglassNet(num_stacks=hg_stacks, num_blocks=hg_blocks, num_classes=hg_num_classes,
                                              batch_norm_momentum=hg_batch_norm_momentum, use_layer_norm=hg_use_layer_norm,
                                              width=width, height=height)
        self.soft_argmax = SoftArgmax2D(window_fn="Parzen")
        self.joint_format_transform = transformer_fn
        self.twod_threed = Transform2D3DNet(linear_size, num_stage, p_dropout, dataset_normlization, input_size, output_size)


    def load(self, file1, file2=None):
        """
        If dir2 == None, then try to load the WHOLE network from the first file.
        Otherwise, try to lead the stacked hourglass network from file1 and the 2d to 3d transformer from file2

        :param file1: Either a checkpoint for a complete stitched network, or, a checkpoint for a stacked hourglass
            network
        :param file2: Either None or a checkpoint for a 2d to 3d pose transformer network.
        :return: Nothing. Sets internal weights/state.
        """
        if file2 is not None:
            hourglass_checkpoint = torch.load(file1)
            twod_threed_checkpoint = torch.load(file2)

            self.stacked_hourglass.load_state_dict(hourglass_checkpoint['state_dict'])
            self.twod_threed.load_state_dict(twod_threed_checkpoint['state_dict'])

        else:
            checkpoint = torch.load(file1)
            self.load_state_dict(checkpoint['state_dict'])


    def move_hip_joint_to_center(self, poses):
        return poses


    def forward(self, x, meta):
        """
        The forward pass. Just apply each of the subnetwork and softargmax in the correct order.

        Meta must be a dictionary of

        :param x: The input, an RGB image
        :param meta: A dictionary containing all of the data required for data normalization/denormalization.
            It must be a dictionary of the following form:
                'center': bounding box centers (that the images were cropped around)
                'scale': a scaling that was applied to the image before
            If using data normalization, we also need the following values:
                '2d_mean': The mean joint values in 2D from the dataset
                '2d_std': The std dev of joint values in 2D from the dataset
        :return: The outputs from the pipeline. The 2D predictions,
            the 2D predictions transformed into another coordinate system (i.e. MPII -> H3.6m)
            and the 3D predictions
        """
        # Unpack meta
        centers = meta['center']
        scales = meta['scale']
        mean_2d = meta['2d_mean'] if '2d_mean' in meta else None
        std_2d = meta['2d_std'] if '2d_std' in meta else None

        # 2D prediction
        heatmaps = self.stacked_hourglass(x)
        final_heatmap = heatmaps[-1]
        twod_preds = self.soft_argmax(final_heatmap)

        # Get the 2D pose in the image
        twod_preds_out = final_preds_post_processing(final_heatmap, twod_preds, centers, scales, [64, 64], True)

        # Compute the input to the 3D baseline network (normalize accordingly)
        # This include re-ordering the joints
        # If using dataset statistics for normalization, we should re-scale the pose as if it were 1000x1002 as in h36m
        # Instance normalization it will not matter the scale, so we can ignore this. (We could also safely multiply by
        # any constant, but that would be confusing).
        twod_preds = self.joint_format_transform(twod_preds)
        if self.baseline_dataset_normalization:
            twod_preds *= 1000.0 / 64.0
        normalized_twod_preds = data_utils.normalize_poses(twod_preds, 16, self.baseline_dataset_normalization,
                                                           pose_mean=mean_2d, pose_std=std_2d, is_2d=True)

        # Run through the 3D baseline network
        threed_preds = self.twod_threed(normalized_twod_preds)

        # Return the heatmaps (for training 2D predictions), 2d predictions (image coords) and normalized 3D prediction
        # Loss uses normalized 3D output, so, we defer de-normalizing 3D poses to later
        return heatmaps, twod_preds_out, threed_preds








