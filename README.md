# ShadowHand eigengrasps repository
This repository provides the tools necessary to extract eigengrasps from the ContactPose dataset.

## Downloading the ContactPose dataset
The dataset can be downloaded from the [ContactPose github repository](https://github.com/facebookresearch/ContactPose).

> **Warning:** The dataset folder is assumed to be in the same folder as this repository. E.g. if sh_eigen is located under /home/user/sh_eigen, the dataset is assumed to be located at /home/user/ContactPose.

## Verifying the data
Some grasps of the ContactPose dataset seem to suffer from bad sensor readings and are distorted. These grasps have to be filtered out manually. The filtered grasps are located in the [hand verification file](/saves/hand_verification.json). You can rerun the verification with

```console
user@pc:~$ python verification.py --task hands
```

## Fitting the ShadowHand to hand poses from the dataset
For each hand pose, 100 optimization runs with our gradient-based inverse kinematics solver are performed. You can rerun the optimization with 

```console
user@pc:~$ python main.py
```

> **Note:** The optimization is heavily parallelized to improve runtimes and will greatly profit from running on CPUs with high core count.

## Verifying the final solutions
In order to verify the optimized grasps, you can visually inspect each grasp by running 

```console
user@pc:~$ python verification.py --task solutions
```

The verified solutions are contained in the [joint verification file](/saves/joint_verification.json).

## Final eigengrasps
The final eigengrasp analysis is done by running the [data analysis notebook](data_analysis.ipynb). Eigengrasps are saved in the [eigengrasp file](/saves/sh_joints.json).
