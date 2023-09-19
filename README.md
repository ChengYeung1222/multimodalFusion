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

## Usage

### Data Preparation

Acquire projected shape descriptors (see [3DMPM](https://github.com/ChengYeung1222/3DMPM)) and their corresponding
coordinates.

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
