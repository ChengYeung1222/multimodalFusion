# multimodalFusion
This repository contains the PyTorch implementation of the multimodal fusion learning method proposed in the paper:

Zheng et al. 2023. Multimodal Learning for Comprehensive 3D Mineral Prospectivity Modeling: Jointly Learned Structure-Fluid Relationships.

## Requirements

* Python 3.6
* PyTorch 1.2.0
* Torchvision 0.4.0
* Visdom 0.1.8
* Cuda92
* Cudnn 10.0.130

## Content

- **custom_data_io.py**: Contains a data generator class for input pipelines.
- **Models.py**: Class with the definition of the multimodal fusion network, which is designed to effectively integrate and align information from both structural and fluid data modalities, leveraging Canonical Correlation Analysis (CCA) to guide the fusion process
- **objectives.py**: Measurements of CCA regularization, which aims to maximize the correlation between the two modalities. 
- **radam.py**: A warmup-mechanism variant of Adam that can rectify the variance of the adaptive learning rate.
- **MultimodalFusion.py**: Script to run the training process of multimodal fusion learning method for comprehensive 3D mineral prospectivity modeling, combining structural and fluid information through a deep network architecture.
