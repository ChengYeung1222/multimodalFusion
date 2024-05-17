# multimodalFusion
This repository contains the PyTorch implementation of the joint integration method proposed in the paper:

_[Deep multimodal fusion for 3D mineral prospectivity modeling: Integration of geological models and simulation data via canonical-correlated joint fusion networks](https://www.sciencedirect.com/science/article/pii/S0098300424001018?via%3Dihub)_, 

now accessible in Computers & Geosciences.

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

## Usage

### Data Preparation

Acquire projected shape descriptors (see [3DMPM](https://github.com/ChengYeung1222/3DMPM)) and their corresponding
properties.

Create `.csv` files for training and validation data. Each of them lists the complete path to your
projected images together with labels (ore-bearing and non-ore-bearing) and standardized fluid properties in the following
structure:

```
Example known_area.csv:
/path/to/known/area/0.bin, 0, _, _, _, z_qx, z_qy, z_qz, SIG1, SSI, VSI, temp_x, temp_y, temp_z
/path/to/known/area/1.bin, 1, _, _, _, z_qx, z_qy, z_qz, SIG1, SSI, VSI, temp_x, temp_y, temp_z
/path/to/known/area/2.bin, 1, _, _, _, z_qx, z_qy, z_qz, SIG1, SSI, VSI, temp_x, temp_y, temp_z 
/path/to/known/area/3.bin, 0, _, _, _, z_qx, z_qy, z_qz, SIG1, SSI, VSI, temp_x, temp_y, temp_z 
.
.
```

here the first, second and sixth to fourteenth columns correspond to paths, labels and fluid properties.

Customize the input and output directories in `MultimodalFusion.py`:

```
source_list = './YOUR_TRAINING_DATA_CSV_PATH'
validation_list = './YOUR_VALIDATION_DATA_CSV_PATH'

source_name = 'known area'
test_name = 'unknown area/validation'

ckpt_path = './YOUR_CKPT_SAVING_PATH'
# restore
ckpt_model = './YOUR_TRAINED_CKPT_PATH'
ckpt_model_mlp = './YOUR_MLP_BRANCH_TRAINED_CKPT_PATH'
```

### Fusion Mode

Change the booleans `parallel = True` and `correlation = True` to switch alignment terms involved.

### Visualization support

The code has Visdom summaries implemented so that you can follow the training progress
by running

```
python -m visdom.server
```

which can start a visdom server on a given port.

### Citation

If you use this library for your research, we would be pleased if you cite the following papers:

```
@article{ZHENG2024105618,
title = {Deep multimodal fusion for 3D mineral prospectivity modeling: Integration of geological models and simulation data via canonical-correlated joint fusion networks},
journal = {Computers & Geosciences},
pages = {105618},
year = {2024},
issn = {0098-3004},
doi = {https://doi.org/10.1016/j.cageo.2024.105618},
url = {https://www.sciencedirect.com/science/article/pii/S0098300424001018},
author = {Yang Zheng and Hao Deng and Jingjie Wu and Shaofeng Xie and Xinyue Li and Yudong Chen and Nan Li and Keyan Xiao and Norbert Pfeifer and Xiancheng Mao},
keywords = {Mineral prospectivity modeling, Multimodal fusion, 3D geological models, Geodynamic simulation data, Canonical correlation analysis},
abstract = {Data-driven three-dimensional (3D) mineral prospectivity modeling (MPM) employs diverse 3D exploration indicators to express geological architecture and associated characteristics in ore systems. The integration of 3D geological models with 3D computational simulation data enhances the effectiveness of 3D MPM in representing the geological architecture and its coupled geodynamic processes that govern mineralization. Despite variations in modality (i.e., source, representation, and information abstraction levels) between geological models and simulation data, the cross-modal gap between these two types of data remains underexplored in 3D MPM. This paper presents a novel 3D MPM approach that robustly fuses multimodal information from geological models and simulation data. Acknowledging the coupled and correlated nature of geological architectures and geodynamic processes, a joint fusion strategy is employed, aligning information from both modalities by enforcing their correlation. A joint fusion neural network is devised to extract maximally correlated features from geological models and simulation data, fusing them in a cross-modality feature space. Specifically, correlation analysis (CCA) regularization is utilized to maximize the correlation between features of the two modalities, guiding the network to learn coordinated and joint fused features associated with mineralization. This results in a more effective 3D mineral prospectivity model that harnesses the strengths from both modalities for mineral exploration targeting. The proposed method is evaluated in a case study of the world-class Jiaojia gold deposit, NE China. Extensive experiments were carried out to compare the proposed method with state-of-the-art methods, methods using unimodal data, and variants without CCA regularization. Results demonstrate the superior performance of the proposed method in terms of prediction accuracy and targeting efficacy, highlighting the importance of CCA regularization in enhancing predictive power in 3D MPM.}
}
```
