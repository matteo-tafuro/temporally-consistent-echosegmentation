# Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation

This repository contains the code for the [MICCAI 2023](https://conferences.miccai.org/2023/en/) submission _"Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation"_. 

<p align="center">
<img src="https://i.ibb.co/cFSL0r6/method-diagram-png.png" width=70%>
</p>

# Table of contents
- [Getting started](#nnu-net)
    - [Environment and requirements](#environment-and-requirements)
    - [Data preparation](#data-preparation)
- 

# Getting started

## Environment and requirements
1. To install the required packages, create a new conda environment using the provided YAML file:

    ```bash
    conda env create -f environment.yaml
    ```

2. Activate the environment:

    ```bash
    conda activate echosegmentation
    ```

3. Since PyTorch is OS- and CUDA-dependent, install `pytorch` and `torchvision` according to your machine. For this, use [light-the-torch](https://github.com/pmeier/light-the-torch), a small utility included in the provided YAML file that auto-detects compatible CUDA versions from the local setup and installs the correct PyTorch binaries without user interference. Use the following command:
    ```bash
    ltt install torch torchvision
    ```

4. TorchIO is used for efficient loading, preprocessing and augmentation of medical images. Since TorchIO should be installed *after* PyTorch, run the following command: 

    ```bash
    pip install torchio
    ```

5. Finally, install nnU-Net as an _integrative framework_. This repository already contains a copy of [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) as of Feb 3, 2023. All you need to do is:
    ```bash
    cd models/nnUNet
    pip install -e .
    ```

This codebase was tested on a machine with Ubuntu 18, two Intel
Xeon Gold 6128 CPUs (6 cores, 3.40GHz) and a GeForce RTX 2080 Ti. The tested version of the machine-dependent packages are specified below:
```yaml
- pytorch=2.0.0
- torchvision=0.15.1
- torchio=0.18.90
```

## Data preparation
The paper uses two public datasets: [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) and [TED](https://humanheart-project.creatis.insa-lyon.fr/ted.html).

### Data download
- Add the CAMUS data to the repository. Download both _training_ and _testing_ data from the [download website](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8), then move the unzipped `training` and `testing` folders to `data/camus`.
- Add the TED data to the repository. Download the _database_ folder from the [download website](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62840fcd73e9f00479084885), then move the unzipped `database` folder to `data/ted`.

The `/data` folder should look like the following:
```bash
data/
├── camus/
│   ├── training/
│   │   ├── patient0001/
│   │   ├── patient0002/
│   │   └── ...
│   └── testing/
│       ├── patient0002/
│       ├── patient0001/
│       └── ...
└── ted/
    └── database/
        ├── patient001/
        ├── patient002/
        └── ...
```


### Data pre-processing
The CAMUS dataset is converted to an HDF5 file for fast I/O. This significantly speeds up the computations and consequently the training of the models. To convert the dataset into HDF5, navigate to the `data/camus/` directory and run the `camus_hdf5_conversion.py` script:

```bash
cd data/camus/
python camus_hdf5_conversion.py
```

# Psuedo-labels generation
To obtain accurate and temporally consistent 3D (2D+time) segmentations from a sparsely labeled dataset, the method first generates the pseudo-labels for those frames that lack reference segmentations. This is done through the iterative application of image registration. Thereafter, the method uses these pseudo-labels to augment sparse reference annotations and train a segmentation model.

## Image registration model
### Training
To train the image registration model, navigate to the `pseudolabels_generation/` directory and run the `train_IR_model.py` script:
```bash
cd pseudolabels_generation/
python train_IR_model.py
```
The script trains a DIRNet for 10,000 epochs, using 32 kernels of size 32×32, a grid spacing of 32 and a regularization hyperparameter value of 1.0 for the bending energy penalty to prevent folding. If you want to experiment with the settings, use `python train_IR_model.py -h` for more information.

Additionally, by default, the script leaves out 25 patients from the training set to run validation. If you want to exclude the TED patients to aid the downstream  evaluation of the pseudo-labels generation, please run `python train_IR_model.py --leave_out_patients -1`.

### Pre-trained models
The weights of the pre-trained models are publicly available [here](https://mega.nz/folder/vQNRHKyL#NdjVjpTxSKaWZA7Sv_I07w). Please download the `LitDIRNet` folder and place it in `pseudolabels_generation/TorchIR/output`.

### Evaluation
The evaluate the image registration results, a qualitative evaluation is conducted on six randomly-chosen patients. One sequence is selected for each view (2CH, 4CH) and each image quality (Poor, Medium, Good). By default, the training script evaluates the following patients at the end of the training procedure:

<p align="center">

| Patient | View | Quality |
| ------- | ---- | ------- |
| 271     | 2CH  | Poor    |
| 437     | 2CH  | Medium  |
| 178     | 2CH  | Good    |
| 13      | 4CH  | Poor    |
| 82      | 4CH  | Medium  |
| 359     | 4CH  | Good    |

</p>

For each of these patients, four outputs are generated:
- *Animation of the propagation*, to assess the propagation of the given masks over time;
- *Comparison between the GT and the last propagated mask*, together with a measure of overlap (DICE score);
- *Average warped image*, along with the absolute difference from the ground truth. The less blurry the average warped image is, the better the registration.
- *Boxplot of the Jacobian Determinant*, to gauge the presence of folding.

## Labels propagation
The trained DIR model is used to propagate the masks over time.
### Inference
Run the `propagate_mask.py` script in `pseudolabels_generation/`. Change the model path before running it, then simply do:
```bash
python propagate_mask.py
```
By default, the script generates the pseudo-labels for _all_ patients in the training set. Feel free to experiment using the provided code.
### Evaluation
