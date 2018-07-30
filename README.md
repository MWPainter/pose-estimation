# CVPR2019 Project

## Repo Structure

Two other git repos are embedded within this repo (and slightly modified). They originally come 
from the repos <https://github.com/bearpaw/pytorch-pose> and 
<https://github.com/weigq/3d_pose_baseline_pytorch>. 

~~~~ 
README.md                                           readme
main.py                                             main entry point for training (scripts)
options.py                                          options to run code with
eval.py                                             entry point for evaluating predicitons (quantitive scores)
vis.py                                              entry point for visualizing results
twod_threed/                                        folder containing code for 2D pose to 3D pose (from: https://github.com/weigq/3d_pose_baseline_pytorch)
    ...
stacked_hourglass/                                  folder containing code for RGB to 2D pose estimation (from: 
    ...                                      
model_checkpoints/                                             folder where models will be saved
    ... (empty initially)
visualizations/                                     folder where visualizations will be saved
    ... (empty initially)                            
data/                                               folder to save data (including outputs from networks)
    ... (empty initially)
~~~~

N.B. we didn't write the code in the "twod_threed" and "stacked_hourglass" folders, so the overall "code consistency" is 
likely to be missing. All code outside of these folders was hopefully written in a consistent manor, and hopefully also 
somewhat documents the other repositories in how they are used.

## Changes made to the pre-existing repos
### Stacked Hourglass code
- Changes to parse_args options, for consistency
- Added `stacked_hourglass/utils/run.py` which contains code to load a trained model, run predictions on some data and save those predictions.

### 2D to 3D code
- Changes to parse_args options, for consistency

## Data

### MPII dataset
Download the images:
`wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz`

Soft link the data to `<REPO_DIR>/stacked_hourglass/data/mpii/images` using the command: `ln -s <MPII_DATA_DIR>/images <REPO_DIR>/stacked_hourglass/data/mpii/images`.


###Human3.6m 2D and 3D pose data
TODO

### Human3.6m Dataset
TODO

### TODO:
TODO: describe the process of downloading the data (incl downloading the 2D and 3D ground truths).

TODO: explain that it needs to be simlinked or copied into the `data` directory.

## Running the code

### Training Models
Models are trained using the following commands:

- `python train.py "hourglass"` Train a stacked hourglass network (RGB image > 2D pose)
    - Prereqs: MPII dataset downloaded as above
    - Default `--data_dir` specifies where the input images of MPII are (default: `./data/hourglass/images`)
    - Example usage (assuming that data has been setup as above) `python main.py hourglass --exp test`
- `python train.py "2d3d"` Train the "3D pose baseline model". (2D pose > 3D pose)
    - Prereqs: Human3.6m 2D and 3D pose data downloaded as above
    - Default `--data_dir` specifies where the 2D pose estimates input is (default: `./data/2d3d/train_2d.pth.tar`)
- `python train.py "stitched"` Train the "Stitched" model
    - `--load_hourglass <model_dir>` Specify a checkpoint to load the hourglass model from (if none specified, then we randomly initialize the network) 
    - `--load_2d3d <model_dir>` Specify a checkpoint to load the 2d3d model from (if none specified, then we randomly initialize the network)
    - `--load <model_dir>` Spefies a checkpoint for the **entire** stitched network to load from 
- The following are options that can be given to ANY of the above commands
    - `--exp <id_str>` Provides an experiment id to append to the name of the output file/folder (string) (default: '0') 
    - `--data_dir <data_dir>` Specify an input dataset (default: depends on script above)
    - `--checkpoint_dir <model_dir>` Specify where model checkpoints are stored (default: `model_checkpoints`)
    - `--output_dir <data_dir>` Specify where to put the output (will be saved in a folder `<output_dir>/<script_<exp id>`)
    - `--load <model_dir>` Specify a checkpoint to load training from 


TODO: explain where models are saved and the file format

TODO: continue explaining params. For example, how to specify the checkpoints for each of the models in the 'stitched' model, and --lr. (Have one per **useful** option in options.py)


### Running Models
Saved models can be run on a dataset to provide predictions:

- `python run.py "hourglass"` Runs the stacked hourglass network to get 2D pose predictions from 
    - `--load <model_dir>` required, filename for the network checkpoint to be used
    - `--data_dir <data_dir>` required, the directory for the images to run the network on
    - `--output_dir <data_dir>` the directory to store the predictions at (in a PyTorch dataset)
    - `--process_as_video` Process videos when using run.py with a network that operates on single frames