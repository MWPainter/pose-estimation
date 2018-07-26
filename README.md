# CVPR2019 Project

## Repo Structure

Two other git repos are embedded within this repo (and slightly modified). They originally come 
from the repos https://github.com/wbenbihi/hourglasstensorlfow and 
https://github.com/weigq/3d_pose_baseline_pytorch. 

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
models/                                             folder where models will be saved
    ... (empty initially)
visualizations/                                     folder where visualizations will be saved
    ... (empty initially)                            
data/                                               folder to save data (including outputs from networks)
    ... (empty initially)
~~~~

Note that we didn't write the code from the other git repos, and will not be maintaining that code.

## Data

TODO: describe the process of downloading the data (incl downloading the 2D and 3D ground truths).

TODO: explain that it needs to be simlinked or copied into the `data` directory.

## Running the code

### Training Models
Models are trained using the following commands:

- `python main.py "hourglass"` Train a stacked hourglass network (RGB image > 2D pose)
    - TODO: `--id` id to use in the saved model
    - TODO: `--dataset` use to specify a dataset directory
    - TODO: models will be saved to `./models/hourglass_<id>/chkpoint/...`
    - TODO: final model will be saved to `./models/hourglass_<id>/final_model`
    - TODO: best model will be saved to `./models/hourglass_<id>/best_model`
    - TODO: training log 
- `python main.py "2d3d"` Train the "3D pose baseline model". (2D pose > 3D pose)
    - TODO: `--id` id to use in the saved model (default: '0')
    - TODO: `--input` specifies where the 2D pose estimates input is (default: './data/2d3d/train_2d.pth.tar')
    - TODO: list all of the options that are provided by the core code + remove lots of options
    - TODO: models will be saved to `./models/hourglass_<id>/chkpoint/...`
    - TODO: final model will be saved to `./models/hourglass_<id>/final_model`
    - TODO: best model will be saved to `./models/hourglass_<id>/best_model`
- The following are options that can be given to ANY of the above commands
    - TODO: things like `--lr` for the learning rate
    - TODO: clean up the options.py file

TODO: something with `--output`, where to write the predictions for example


TODO: finish descriptions of code as go

TODO: same for eval 