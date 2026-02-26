# On Deepfake-Powered C-Level Fraud: Risks, Attacks and Mitigation

This repository contains code used for my Master Project from [EPFL](https://epfl.ch) done at [Swiss Post](https://post.ch) on deepfake detection in video-conferencing.

If you wish to contact me, you can open an issue on this GitHub repository.

The code for this project heavily builds on top of [DeepFakeBench](https://github.com/SCLBD/DeepfakeBench) by Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu and [DF40](https://github.com/YZY-stack/DF40) by Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu, Donghao Luo, Chengjie Wang, Shouhong Ding, Yunsheng Wu, Li Yuan.

The dataloader implementation is theirs, as well as the runner for the models.

## Relevant Files and Directories

- [opti_rt_detection.ipynb](opti_rt_detection.ipynb) contains a proof-of-concept of a fusion of deepfake detectors in real-time from a screen-capture
- [poc_rt_detection.ipynb](poc_rt_detection.ipynb) is a older version of the same POC with input from the webcam
- [requirements.txt](requirements.txt) contains the conda environement information used to run this project
- the [exps/](exps/) directory contains results from benchmarks on different datasets
- [benchmarks.ipynb](benchmarks.ipynb) is a general test runner anmd logger


## Model weights, datasets and meta-information, extensive experiments logs

As all of those files are too heavy for Github, please open an issue if you wish to access to them. Please note that some of these are not my property, thus in some cases I won't be able to distribute them.

