# CVPR2019 Project

## Repo Structure

Two other git repos are embedded within this repo (and slightly modified). They originally come 
from the repos [Stacked Hourglass (Pytorch)](https://github.com/bearpaw/pytorch-pose) and 
[3D Baseline (PyTorch)](https://github.com/weigq/3d_pose_baseline_pytorch). We also use a few files from 
[3D Baseline (Tensorflow)](https://github.com/una-dinosauria/3d-pose-baseline).

Below we outline the structure of the repository. We don't recursively outline the prior repos, as 
they contain their own (untouched) readme's.

~~~~ 
README.md                                           readme
train.py                                            main entry point for training (scripts)
run.py                                              main entry point for just running the networks for inference/forward pass. (scripts)
viz.py                                              main entry point for visualization, producing image files (scripts)
eval.py                                             main entry point for evaluation code (scripts)
options.py                                          options to run code with
twod_threed/                                        folder containing code for 2D pose to 3D pose. The "3D Pose Baseline". (from: https://github.com/weigq/3d_pose_baseline_pytorch)
    ...
stacked_hourglass/                                  folder containing code for RGB to 2D pose estimation (from: 
    ...                                      
generative_models/                                  folder containing all code for a 3D pose WGAN
    base_network.py                                     defines a fully connected resnet (a base network used for the generative models)
    fc_gan.py                                           defines a WGAN that uses the fully connected resnet defined in "./base_network.py".
    train.py                                            defines all subroutines to use the training loop defined in 'utils/train_utils.py' and puts them together in a training function/script
    viz.py                                              defined functions/scripts that can be used to visualize the 3D pose WGAN
utils/                                              folder containing all code that is useful across multiple librarys. (I.e. the 'utility functions').
    camera_utils.py                                     utility functions for logic related to cameras and projections (originally taken from 3D Baseline (Tensorflow)).
    data_utils.py                                       utility functions for logic related to data processing, such as normalization (originally taken from 3D Baseline (Tensorflow)). (Added logic for orthogonal data augmentation).
    human36m_dataset.py                                 a general human3.6m dataset, that provides general quadruples, for the I/O of both the stacked hourglass and 3D baseline networks (includes logic for orthogonal data augmentation).
    osutils.py                                          os utils (saving files/makedirs etc)
    plotting_utils.py                                   utility functions that keep averages and compute things that we would often like to plot/monitor during training. 
    training_utils.py                                   defines a generic parameterised training loop script. (One can implemnt 5 functions and then have training loop code).
    transform.py                                        defines functions for converting between pose representations (for example, convert MPII 2D Poses to Human36m 2D Poses). So really this just re-orders joints in poses. 
stitched/                                           Contains all logic regarding 
    train.py                                            defines all subroutines to use the training loop (for fine tuning the network) defined in 'utils/train_utils.py' and puts them together in a training function/script 
    run.py                                              scripts to run the network (foward pass)
    soft_argmax.py                                      defines a PyTorch nn.Module implementing a 'soft argmax' function
    stitched_network.py                                 defines a network (PyTorch nn.Module) combining twod_threed and stacked_hourglass models together 
model_checkpoints/                                  folder where models will be saved
    ... (empty initially)
visualizations/                                     folder where visualizations will be saved (by default)
    ... (empty initially)                            
data/                                               folder to save data (including outputs from networks)
    ... (empty initially)
tb_logs/                                            folder to save tensorboard logs to
    ... (empty initially)
.cache/                                             folder used to save cached statistics about datasets (for normalizing data)
    ... (empty initially)
~~~~

N.B. The "stacked_hourglass" folder contains all of the code from [Stacked Hourglass (Pytorch)](https://github.com/bearpaw/pytorch-pose)
and "twod_threed" folder contains all of the code from [3D Baseline (PyTorch)](https://github.com/weigq/3d_pose_baseline_pytorch). 
Additionally, "utils/osutils.py" was pulles from one of the two repos, and "utils/camera_utils.py" and "utils.data_utils.py" 
are taken from [3D Baseline (TensorFlow)](https://github.com/una-dinosauria/3d-pose-baseline).

Overall we try to maintain a consistent style, however, there will be inconsistencies from the previous code bases that 
have been merged. We have added some comments and refactored some parts of their code where necessary to 
implement the changes we wanted. 


## Changes made to the pre-existing repos
### Stacked Hourglass code
- Original repo from [https://github.com/bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose)
- Changes to parse_args options, for consistency, and move options logic to the root directory in 'options.py'.
- Added `stacked_hourglass/utils/run.py` which contains code to load a trained model, run predictions on some data and save those predictions.
- Cannabilized the `visualize` function in `stacked_hourglass/evaluation/utils.py` to provide a clean visualization of a 2d pose overlay on some image.
- Updated the MPII dataset to be able to set and get the mean and stddev for data normalization. Changes found in dataset 
    `stacked_hourglass/pose/dataset/mpii.py` and the training loop in `stacked_hourglass/example/mpii.py`
- (Incomplete). Added option for data augmentation in dataset, independent of the "train" flag. (Sometimes we would like to turn  off 
    the data augmentation on the train set. Train flag is necessary as MPII anotations are all in a single file). Dataset 
    class found in `stacked_hourglass/pose/dataset/mpii.py`.
- Added option to add attention between the lowest dimensional representations in the hourglass modules. Changes are in `stacked_hourglass/pose/models/hourglass.py`
- Copied the mpii training code in `stacked_hourglass/example/mpii.py` to `stacked_hourglass/example/eva.py` and altered it to use the temporal data given by the EVA dataset (i.e. each example is a sequence of some number of frames).
- Altered `stacked_hourglass/example/mpii.py` to reflect (and ignore using `NULL` values) changes made for attention in hourglass model.
- Added [TensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training curves at runtime.
- (Re) Added caching of dataset statistics, defaulting to the `.cache` directory.
- Factored out data augmentation in the MPII dataset to be specified using a seperate option than if it's the training/validation set. So that we can visualize the training set properly.
- Added optional random masking to MPII dataset. Used in `stacked_hourglass/pose/dataset/mpii.py` and implemented in `stacked_hourglass/pose/utils/transforms.py`, and theresfore the Human Eva I dataset too (as that was copied). 
- Added code to train using multiple GPUs using [Horovod](https://github.com/uber/horovod) 
- Added gradient clipping
- Commented out unused code (for clarity that we're completely not using some files).

### 2D to 3D code
- Original repo from [https://github.com/weigq/3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- Changes to parse_args options, for consistency
- Added `twod_threed/run.py` which contains code to load a trained model, run predictions on some data and save those predictions.
- Added processing to handle input shapes (batch_size, num_joint, 2) rather than just (batch_size, 2*num_joints). In 
    such a case, output is of shape (batch_size, num_joints, 3). Changes are found in `twod_threed/src/model.py`'s 
    forward function. 
- Used vizualisation code from the original tensorflow implmentation: [https://github.com/una-dinosauria/3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline).
    - Copied files to `twod_threed/src/viz.py` and `twod_threed/src/data_utils.py`
    - Both files are edited to work directly out of stacked hourglass/MPII format of joint co-ordinates.
    - Renamed `camera.py` to `camaras.py` in `twod_threed/src` to be consistent with files copied from the tensorflow implementation.
- Factored out code in the Human36m dataset class that translated keys for the 2D data to the keys for the 3D data. 
    (Still unsure why they couldn't just be the same...). File `twod_threed/src/datasets/human36m.py`
- Added [TensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training curves at runtime. 
- Added code to train using multiple GPUs using [Horovod](https://github.com/uber/horovod) 
- Added options to control gradient clipping
- Commented out unused code (for clarity that we're completely not using some files).
- Changed the dataset to just be an interface for the `utils/human36m_dataset.py` dataset.
- (Commented out mostly). Added code for a Cycle GAN. Changes made to `twod_threed/src/main.py` for the training loop (to 
    add GAN losses and Cycle consistency losses). Changes also made to `twod_threed/src/model.py`, that implements a PyTorch 
    nn.Module for a (weak) perspective transform. 

## Data

### [MPII dataset](http://human-pose.mpi-inf.mpg.de/)
Download the images:
`wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz`

Soft link the data to `<REPO_DIR>/stacked_hourglass/data/mpii/images` using the command: `ln -s <MPII_DATA_DIR>/images <REPO_DIR>/stacked_hourglass/data/mpii/images`.


### [Human3.6m Dataset](http://vision.imar.ro/human3.6m/description.php)
TODO

## Running the code
### Training Models (using train.py)
Models are trained using the following commands:

- The following are options that can be given to ANY of the commands below
    - `--workers` The number of workers to use in the PyTorch DataLoader objects. (This is **very important** as training is significantly bottlenecked by the speed of data loading and training can be orders of magnitude slower without this set to >1 (default is 6)).
    - `--use_amsprop` To train with the [AMSProp](https://openreview.net/pdf?id=ryQu7f-RZ) optimizer
    - `--lr` Specify a specific learning rate
    - `--exp` An experiment id. This will be used for naming the checkpoint files and so on.
    - `--tb_dir` A location to put Tensorboard summaries at.
    - `--seed` Specify a seed to use for random number generation
    - `--load` The (directory of a) model checkpoint use to restart training.
    - `--checkpoint_dir` A directory to save checkpoints to. If this argument is `DIR` then models will be saved in the folder `DIR/hourglass_mpii_<EXP_ID>/`, where `<EXP_ID>` is the experiment id defined with `--exp`.
    - `--data_dir` The location of the training data.  
    - `--epochs` The number of epochs to use in training
- `python train.py "hourglass_mpii"` Train a stacked hourglass network (RGB image > 2D pose) on the MPII dataset
    - Prereqs: MPII dataset downloaded as above
    - `--stacks` The number of hourglasses to stack
    - `--features` The number of features to learn at every stack
    - `--blocks` The number of residual modules in each residual block of the hourglasses
    - `--remove_intermediate_supervision` Option to turn off the use of intermediate supervisions.
    - `--scale_weight_factor` Scale all weights by some factor (at initialization).
    - `--use_layer_norm` Use layer norm rather than using batch normalization.
    - `--no_batch_norm_affine` Remove the affine transform for the output of batch norm layers.
    - `--predict_joint_visibility` If we should simultaneously train a joint visibility network.
    - `--joint_visibility_loss_coeff` The coefficient used infront of the loss function for joint visibility (as part of the whole loss function)
    - `--batch_norm_momentum` Specify the momentum term in batch norm layers of the network.
    - `--augment_training_data` Defaults to true, but can be set to false, to standard data augmentation of training data.
    - `--add_random_masking` Use to add random masking over the training data. (Independent to augmentating the training data, so `--augment_training_data False --add_random_masking` will add random masking without random rotating and scaling).
    - `--mask_prob` In random masking, the probability of adding a mask.
    - `--orientation_prob` Specifies the probability of the random mask being a vertical bar over the image (rather than horizontal).
    - `--mean_valued_prob` The probability that we mask with a solid block with mean pixel value, (otherwise we add random noise for the mask).
    - `--max_cover_ratio` Specify the maximum ratio of the width/height of a person(s bounding box) that we allow to be covered by the mask. 
    - `--noise_std` The stddev of the Gaussian noise if the mask consists of random noise.
    - The default options are equivelent to running the following command `python train.py hourglass_mpii --checkpoint_dir model_checkpoints/ --exp default --tb_dir tb_logs/`
- `python train.py "2d3d_h36m"` Train the "3D pose baseline model", on the Human3.6m dataset. (2D pose > 3D pose)
    - Prereqs: Human3.6m data downloaded as above
    - `--orthogonal_data_augmentation` Apply a (random) orthogonal data augmentation to poses in training
    - `--flip_prob` The probability to perform a flip in the orthogonal data augmentation
    - `--z_rotations_only` Use to restrict all rotations to be about the z-axis in the random orthogonal data augmentation
    - `--dataset_normalization` The option to use a normalization over the training dataset statistics, rather than the (default) instance normalization.    
    - The default options are equivelent to running the following command `python train.py hourglass_mpii --checkpoint_dir model_checkpoints/ --exp default --tb_dir tb_logs/`
- `python train.py 3d_pose_gan` Train a WGAN for 3D poses.
    - Prereqs: Human3.6m data downloaded as above.
    - The default options are equivelent to running the following command `python train.py 3d_pose_gan --checkpoint_dir model_checkpoints/ --exp default --tb_dir tb_logs/`
    


### Running Models (forward inference/using run.py)
Saved models can be run on a dataset to provide predictions:

- Any options that altered the architecture during training need to also be set now. 
- The following are options that can be given to ANY of the commands below:
    - `--load <model_dir>` The directory for which to load the model weights from
    - `--data_dir <data_dir>` required, the directory for the images to run the network on
    - `--output_dir <data_dir>` the directory to store the predictions at
- `python run.py hourglass_mpii` Runs the stacked hourglass network to get 2D pose predictions from RGB images. Requirement on MPII dataset + output will be in MPII's joint format.
    - (no specific options)
- `python run.py 2d3d_h36m` Runs the "3D Pose Baseline Model" on some 2D predictions. Dependence here is on Human3.6m dataset objects and using Human3.6m joint formats (different to MPII's joint format).
    - Use all options as in "hourglass_mpii" script
    - `--process_as_video` Process videos when using run.py with a network that operates on single frames
- `python run.py stitched_mpii` Runs the hourglass and 3D baseline networks, stiched together using soft argmax
    - `--load_hourglass` Specify a checkpoint file to load the stacked hourglass network from
    - `--load_2d3d` Specify a checkpoint file to load the 3D baseline network from (for predicting 3D poses from 2D poses)
    - `--load` This option for this particular script will override load_hourglass and load_2d3d
    
    
    
### Visualizing Models (using viz.py)
Once we have processed data in a consistent manor, we can visualize the outputs (and ground truths) using viz.py

Note for this file we have dropped the dataset dependence between the script names and datasets. It should be implicit that data is of the same format as what the network was trained on however.

- `python viz.py 2d_overlay_3d_pred` Outputs visualizations of 2D poses ontop of the original images + adds a 3D pose estimation netxt to it
    - `--img_dir` The direcory for the images input
    - `--2d_pose_estimations` 2d pose estimations for the images input
    - `--3d_pose_estimation` 3d pose estimations for the images input
    - `--output_dir` The directory to output the vizualized images to
- `python viz.py 2d_overlay_3d_gt_3d_pred` Outputs visualizations of 2D poses ontop of the original image, with ground truth 3D pose and estimated 3D pose alongside it   
    - `--img_dir` The direcory for the images input
    - `--2d_pose_estimations` 2d pose estimations for the images input
    - `--3d_pose_ground_truths` 3d pose ground truths for each image
    - `--3d_pose_estimation` 3d pose estimations for the images input
    - `--output_dir` The directory to output the vizualized images to
- `python viz.py 2d_overlay` Just visualized the 2D pose estimations ontop of the original images (visualizations identical to from original stacked hourglass)
    - `--img_dir` The direcory for the images input
    - `--2d_pose_estimations` 2d pose estimations for the images input
    - `--output_dir` The directory to output the vizualized images to 
- `python viz.py 2d_gt_3d_gt_3d_pred` Visualize the 2D and 3D ground truths next to the 3D prediction (visualizations identical to the original 2D3D code)
    - `--2d_pose_ground_truths` 2d pose ground truths for each image
    - `--3d_pose_ground_truths` 3d pose ground truths for each image
    - `--3d_pose_estimation` 3d pose estimations for the images input
    - `--output_dir` The directory to output the vizualized images to
- `python viz.py saliency_and_prob_maps` Visualizes saliency and probability/heat maps from the stacked hourglass network. This will produce images with many rows, one for each joint being predicted.  
    - This script uses the model directly, any options that altered the architecture during training also need to be added now.
    - `--img_dir` The directory for the images to run through the network
    - `--load` The checkpoint file to load the stacked hourglass network from
    - `--output_dir` The directory to save the visualized images
- `python viz.py saliency_and_prob_maps_with_skeleton` The same visualization as 'saliency_and_prob_maps', but with the 
    - All the options from above (and no additional options are needed).
- `python viz.py orthog_augmentation` Visualizes the 2D ground truth, 3D ground truth and 3D prediction from the 3D baseline network on a single example augmented many times using the orthogonal data augmentation.
    - This script uses the model directly, any options that altered the architecture during training also need to be added now.
    - `--z_rotations_only` Use to restrict all rotations to be about the z-axis in the random orthogonal data augmentation
    - `--data_dir` The directory which stores the dataset
    - `--load` The checkpoint file for the model
    - `--index` The index into the dataset that we want to augment and visualize
    - `--num_orientations` The number of augmentations/orientations that we want to visualize
    - `--dataset_normalization` Specifies that the network was trained using dataset normalization, and that we should use that normalization scheme here.
    - `--output_dir` The output to save the visualized images.  
    
  
### Evaluating Models (using eval.py)
Some of our scripts will produce files storing the predictions made on the validation set. We can use this script to 
evaluate the performance on the validation set.

- `python eval.py mpii_PCKh` Will compute PCKh scores based on a prediction file, prints out the scores per joint for PCKh@0.5, and produces PCKh curves per joint (including mean). If multiple files are specified, we plot the different 'models' on the same graph.
    - `--prediction_files` A space seperatred list of prediction files (output by the hourglass_mpii training script)
    - `--model_names` A space seperated list of model names, to be used in the graphs plotted (the ith name should correspond to the model name for the ith prediction file)
    - `--output_dir` A directory to output all of the visualizations.