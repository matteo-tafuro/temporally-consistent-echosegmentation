# Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation

This repository contains the code for the [MICCAI 2023](https://conferences.miccai.org/2023/en/) submission _"Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation"_. 

<p align="center">
<img src="https://github.com/matteo-tafuro/temporally-consistent-echosegmentation/blob/main/images/method-diagram.png?raw=true" width=70%>
</p>

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