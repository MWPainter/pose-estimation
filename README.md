# CVPR2019 Project

## Repo Structure

Two other git repos are embedded within this repo (and slightly modified). They originally come 
from the repos [https://github.com/bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose) and 
[https://github.com/weigq/3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch). 

~~~~ 
README.md                                           readme
main.py                                             main entry point for training (scripts)
options.py                                          options to run code with
transform.py                                        batch processing to change poses in one dataset's joint formate to another (largely unoptimized currently)
eval.py                                             entry point for evaluating predicitons (quantitive scores)
vis.py                                              entry point for visualizing results
twod_threed/                                        folder containing code for 2D pose to 3D pose (from: https://github.com/weigq/3d_pose_baseline_pytorch)
    ...
stacked_hourglass/                                  folder containing code for RGB to 2D pose estimation (from: 
    ...                                      
stitched/                                          
    soft_argmax.py                                  defines a PyTorch nn.Module implementing a 'soft argmax' function
    stitched_network.py                             defines a network (PyTorch nn.Module) combining twod_threed and stacked_hourglass models together 
    ...
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

N.B. we didn't write the code in the "twod_threed" and "stacked_hourglass" folders, so overall "code consistency" will 
be lacking. All code outside of these folders is hopefully written in a consistent manor, and hopefully also 
documents the nested repositories via how they are used. (README.md files for the nested repositories were not touched).

## Changes made to the pre-existing repos
### Stacked Hourglass code
- Original repo from [https://github.com/bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose)
- Changes to parse_args options, for consistency
- Added `stacked_hourglass/utils/run.py` which contains code to load a trained model, run predictions on some data and save those predictions.
- Cannabilized the `visualize` function in `stacked_hourglass/evaluation/utils.py` to provide a clean visualization of a 2d pose overlay on some image.
- Updated the MPII dataset to be able to set and get the mean and stddev for data normalization. (Otherwise we can't train 
    multiple models with different datasets correctly). Changes found in dataset `stacked_hourglass/pose/dataset/mpii.py` 
    and the training loop in `stacked_hourglass/example/mpii.py`
- Added option for data augmentation in dataset, independent of the "train" flag. (Sometimes we would like to turn  off 
    the data augmentation on the train set. Train flag is necessary as MPII anotations are all in a single file). Dataset 
    class found in `stacked_hourglass/pose/dataset/mpii.py`.
- Added a dataset file for the Human_Eva_I dataset, found at `stacked_hourglass/pose/datasets/eva.py`
- Added option to add attention between the lowest dimensional representations in the hourglass modules. Changes are in `stacked_hourglass/pose/models/hourglass.py`
- Copied the mpii training code in `stacked_hourglass/example/mpii.py` to `stacked_hourglass/example/eva.py` and altered it to use the temporal data given by the EVA dataset (i.e. each example is a sequence of some number of frames).
- Altered `stacked_hourglass/example/mpii.py` to reflect (and ignore using `NULL` values) changes made for attention in hourglass model.
- Added [TensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training curves at runtime.
- (Re) Added caching of dataset statistics, defaulting to the `.cache` directory.
- Factored out data augmentation in the MPII dataset to be specified using a seperate option than if it's the training/validation set. So that we can visualize the training set properly.
- Added optional random masking to MPII dataset. Used in `stacked_hourglass/pose/dataset/mpii.py` and implemented in `stacked_hourglass/pose/utils/transforms.py`, and theresfore the Human Eva I dataset too (as that was copied). 
- Added code to train using multiple GPUs using [Horovod](https://github.com/uber/horovod) 

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

## Data

### [MPII dataset](http://human-pose.mpi-inf.mpg.de/)
Download the images:
`wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz`

Soft link the data to `<REPO_DIR>/stacked_hourglass/data/mpii/images` using the command: `ln -s <MPII_DATA_DIR>/images <REPO_DIR>/stacked_hourglass/data/mpii/images`.


###Human3.6m 2D and 3D pose data
TODO

### [Human3.6m Dataset](http://vision.imar.ro/human3.6m/description.php)
TODO

### [Human Eva I dataset](http://humaneva.is.tue.mpg.de/)
TODO

## Running the code

### Training Models
Models are trained using the following commands:

- `python train.py "hourglass_mpii"` Train a stacked hourglass network (RGB image > 2D pose) on the MPII dataset
    - Prereqs: MPII dataset downloaded as above
    - `--checkpoint_dir` A directory to save checkpoints to. If this argument is `DIR` then models will be saved in the folder `DIR/hourglass_mpii_<EXP_ID>/`, where `<EXP_ID>` is the experiment id defined with `--exp`.
    - `--load` The (directory of a) model checkpoint use to restart training.
    - `--exp` An experiment id
    - `--tb_dir` The directory where to store the tensorboard summaries. The tensorboard log directory to use will be `<TB_DIR>/hourglass_mpii_<EXP_ID>_tb_log/`, where `<TB_LOG` is the directory specified with this option.
    - `--remove_intermediate_supervision` Removes intermediate supervisions (i.e. there is only a loss at the ouptput of the final hourglass module)
    - `--adjust_batch_norm` Move batch norm to only be applied immediately before an activation function (and nowhere else)
    - TODO: add options for specifying the number of stacks and blocks in hourglass etc
    - TODO: add options for the changes that we've made (e.g. using attention, but needs to be with a dataset that supports eva)
    - TODO: add all of the options for the data augmentation. `--augment_training_data` and `--no_random_masking` and `--orientation_prob` and so on.
    - The default options are equivelent to running the following command `python train.py hourglass_mpii --checkpoint_dir model_checkpoints/ --exp default -- tb_dir tb_logs/`
- `python train.py "hourglass_eva"` Train a stacked hourglass network (RGB image > 2D pose) on the Human_EVA_I dataset
    - Prereqs: Human_Eva_I dataset downloaded as described above.
    - All options from "hourglass_mpii" work
    - `--add_attention` Option specified will add an attention mechanism to the stacked hourglass networks (at the lowest spatial dimension in the hourglass)
    - `--history_lenth` The history length/window length to use with attention in the time domain.
    - `--dataset_time_series_length` The length of sequences to provide from the dataset. If the dataset length is 100, and history length is 50, then training will slide a window of length 50 over these 100 frames, zero padding appropriately so that there are 100 updates for each minibatch from the dataset.
- `python train.py "2d3d_h36m"` Train the "3D pose baseline model", on the Human3.6m dataset. (2D pose > 3D pose)
    - Prereqs: Human3.6m 2D and 3D pose data downloaded as above
    - `--checkpoint_dir` A directory to save checkpoints to. If this argument is `DIR` then models will be saved in the folder `DIR/hourglass_mpii_<EXP_ID>/`, where `<EXP_ID>` is the experiment id defined with `--exp`.
    - `--load` The (directory of a) model checkpoint use to restart training.
    - `--exp` An experiment id
    - `--tb_dir` The directory where to store the tensorboard summaries. The tensorboard log directory to use will be `<TB_DIR>/hourglass_mpii_<EXP_ID>_tb_log/`, where `<TB_DIR>` is the directory specified with this option.
    - The default options are equivelent to running the following command `python train.py hourglass_mpii --checkpoint_dir model_checkpoints/ --exp default -- tb_dir tb_logs/`
- `python train.py "stitched_eva"` Train the "Stitched" model on the Human Eva I dataset
    - `--load_hourglass <model_dir>` Specify a checkpoint to load the hourglass model from (if none specified, then we randomly initialize the network) 
    - `--load_2d3d <model_dir>` Specify a checkpoint to load the 2d3d model from (if none specified, then we randomly initialize the network)
    - `--load <model_dir>` Spefies a checkpoint for the **entire** stitched network to load from 
    - TODO: Finish options (should be the same as "hourglass_mpii")
    - TODO: Finish with default option values/example
- The following are options that can be given to ANY of the above commands
    - `--workers` The number of workers to use in the PyTorch DataLoader objects. (This is **very important** as training is significantly bottlenecked by the speed of data loading and training can be orders of magnitude slower without this set to >1 (default is 6)).
    - `--use_amsprop` To train with the [AMSProp](https://openreview.net/pdf?id=ryQu7f-RZ) optimizer
    - TODO: describe general training parameters, such as `--lr`
    - TODO: Direct them to `options.py` for their default values.

### Running Models
Saved models can be run on a dataset to provide predictions:

- `python run.py "hourglass_mpii"` Runs the stacked hourglass network to get 2D pose predictions from RGB images. Requirement on MPII dataset + output will be in MPII's joint format.
    - `--load <model_dir>` required, filename for the network checkpoint to be used
    - `--data_dir <data_dir>` required, the directory for the images to run the network on
    - `--output_dir <data_dir>` the directory to store the predictions at (in a PyTorch dataset)
- `python run.py "2d3d_h36m"` Runs the "3D Pose Baseline Model" on some 2D predictions. Dependence here is on Human3.6m dataset objects and using Human3.6m joint formats (different to MPII's joint format).
    - Use all options as in "hourglass_mpii" script
    - `--process_as_video` Process videos when using run.py with a network that operates on single frames
    
    
    
### Visualizing Models
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
  

### Transforming Data
TODO: describe using transform.py