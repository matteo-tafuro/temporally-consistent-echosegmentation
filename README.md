# Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation

This repository contains the code for the [MICCAI 2023](https://conferences.miccai.org/2023/en/) submission _"Temporally consistent segmentations from sparsely labeled echocardiograms using image registration for pseudo-label generation"_. 

<p align="center">
<img src="https://i.ibb.co/cFSL0r6/method-diagram-png.png" width=70%><br>
    <i> Figure 1: The proposed DIR-based pseudo-labels generation method. The provided segmentations are propagated from ED to ES (a) and from ES to ED (e). The masks from the two directions are subsequently aggregated and weighted according to a sinusoidal function (b and d).</i>
</p>


# Table of contents

- [Getting started](#getting-started)
    - [Environment and requirements](#environment-and-requirements)
    - [Data preparation](#data-preparation)
        - [Data download](#data-download)
        - [Data pre-processing](#data-pre-processing)
- [Psuedo-labels generation](#psuedo-labels-generation)
    - [Image registration model](#image-registration-model)
        - [Training](#training)
        - [Pre-trained models](#pre-trained-models)
        - [Evaluation](#evaluation)
    - [Labels propagation](#labels-propagation)
        - [Inference](#inference)
        - [Evaluation](#evaluation)
- [Segmentation](#segmentation)
    - [Training](#training)
        - [Framwork preparation](#framwork-preparation)
        - [2D Sparse nnU-Net](#2d-sparse-nnu-net)
        - [2D Dense nnU-Net](#2d-dense-nnu-net)
        - [3D Dense nnU-Net](#3d-dense-nnu-net)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
        - [Segmentation results: geometric accuracy, LV volume and EF estimation](#segmentation-results-geometric-accuracy-lv-volume-and-ef-estimation)
        - [Temporal smoothness](#temporal-smoothness)
- [Results](#results)


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
To compare the pseudo-labels quality in terms of geometric metrics (as seen in the boxplots of Fig.2 from the paper), run the `evaluate_propagated_masks.py` script in `pseudolabels_generation/`:
```bash
python evaluate_propagated_masks.py
```
However, the comparison includes the pseudo-labels generated by a traditional segmentation model, i.e. the nnU-Net. Therefore, before running this evaluation, you should train the _2D Sparse nnU-Net_ as seen in the [corresponding section](#training).

# Segmentation
The reference segmentations of the echocardiograms are augmented with the pseudo-labels to provide densely labeled reference sequences. This enables the training of 3D (2D+time) segmentation models, which are designed to be trained on densely annotated data. By encoding the time dimension as the third dimension in convolutional space, a 3D model can learn spatiotemporal features that encourage temporally smooth predictions.

## Training
Three models are trained and compared: A 3D nnU-Net is trained on the augmented dataset (_3D Dense nnU-Net_), a 2D nnU-Net trained on the sparsely labeled CAMUS dataset (_2D sparse nnU-Net_) and a 2D nnU-Net trained on the augmented CAMUS dataset (_2D Dense nnU-Net_).

### Framwork preparation

#### Environment variables
First, the nnU-Net requires some environment variables to be set. Navigate to the `segmentation/` directory, then type the following in your terminal:

```bash
export nnUNet_raw_data_base="./data/nnUNet_raw_data_base"
export nnUNet_preprocessed="./data/nnUNet_preprocessed"
export RESULTS_FOLDER="./trained_models"
```

#### Data conversion
nnU-Net expects datasets in a structured format. This format closely (but not entirely) follows the data structure of
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). A conversion script is provided in `segmentation/nnUNet/data/camus_MSD_conversion.py`. Adjust the parameters according to the model and run it. For more information about the task names and IDs, read the `readme.md` file in `segmentation/nnUNet/`. Here we assume that the 2D Sparse, 2D Dense and 3D Dense models have ID 570, 571 and 572 respectively.

### 2D Sparse nnU-Net
1. Before training, nnU-Net requires the _Experiment planning and preprocessing_ step. In your terminal, run:
    ```bash
    nnUNet_plan_and_preprocess -t 570 -pl3d None --verify_dataset_integrity
    ```

2. Train the model. Run:
    ```bash
    nnUNet_train 2d nnUNetTrainerV2 570 X --npz
    ```
    5 different times, for `X=[0,1,2,3,4]`.

3. Find the best nnU-Net configuration:
    ```bash
    nnUNet_find_best_configuration -m 2d -t 570
    ```

### 2D Dense nnU-Net
1. Before training, nnU-Net requires the _Experiment planning and preprocessing_ step. In your terminal, run:
    ```bash
    nnUNet_plan_and_preprocess -t 571 -pl3d None --verify_dataset_integrity
    ```

2. Train the model. Run:
    ```bash
    nnUNet_train 2d nnUNetTrainerV2 571 X --npz
    ```
    5 different times, for `X=[0,1,2,3,4]`.

3. Find the best nnU-Net configuration:
    ```bash
    nnUNet_find_best_configuration -m 2d -t 571
    ```

### 3D Dense nnU-Net
1. Before training, nnU-Net requires the _Experiment planning and preprocessing_ step. In your terminal, run:
    ```bash
    nnUNet_plan_and_preprocess -t 572 --verify_dataset_integrity
    ```

2. Train the model. Run:
    ```bash
    nnUNet_train 3d_fullres nnUNetTrainerV2 572 X --npz
    ```
    5 different times, for `X=[0,1,2,3,4]`.

3. Find the best nnU-Net configuration:
    ```bash
    nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t 572
    ```

## Inference
1. 2D Sparse model:
   ```bash
    nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t TASK_NAME
    ```

2. 2D Dense model:
   ```bash
    nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t TASK_NAME
    ```

3. 3D Dense model:
   ```bash
    nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t TASK_NAME
    ```

## Evaluation
The predicted segmentations are intrinsically evaluated by overlap and boundary metric. Additionally, the segmentation models are evaluated extrinsically on the EF and LV volumes. To aggregate dataset-level statistics for these estimations, the correlation coefficient, bias and mean absolute error (MAE) are calculated between the reference and automatically obtained values. Finally, the temporal consistency of the automatic segmentation is assessed by tracking the area of a given class over time. The smoothness of a sequence is computed as the integral of the second derivative of the resulting curve, which will be denoted as area curve. To account for changes in the slope of the area curve and prevent the loss of information due to opposite bending, the second derivative is squared prior to integration.

### Segmentation results: geometric accuracy, LV volume and EF estimation
The evaluation of the segmentation results is evaluated on the ED and ES frames using the CAMUS submission platform. The raw nnU-Net outputs need to be converted into MHD files that can be correctly processed by the platform. Please use the `process_outputs.py` and upload the resulting MHD files on the submission website.

### Temporal smoothness
The temporal smoothness is defined as the integral of the squared second derivative:
$$
\int_a^b \left(f''(x)\right)^2 dx
$$. 
To replicate Figure 3 and 4 of the paper, please run the `evaluate_smoothness.py` script in the `segmentation` directory.

# Results
The following figure shows the performance of the pseudo-label generation using the forward, backward and bidirectional approach. For comparison, pseudo-labels were also generated using a SoTA 2D nnU-Net trained on the original sparsely labeled CAMUS dataset (2D Sparse nnU-Net), to allow a comparison between our DIR-based pseudo-labels and the predictions of a segmentation model. The analysis of the pseudo-label quality revealed the benefits of bidirectional propagation:
<p align="center">
<img src="https://i.ibb.co/qgtPkQn/pseudolabels-evaluation.png" alt="pseudolabels-evaluation" width=95% /><br>
    <i> Figure 2: Comparison of the pseudo-labels quality in terms of geometric metrics.</i>
</p>

The table below displays the segmentation results at ED and ES. It demonstrates that exploiting the pseudo-labels retains or improves the performance of the model trained on the sparsely labeled dataset, thereby endorsing their quality for downstream applications. In fact, the overlap and boundary metrics show that all three evaluated models perform \textit{at least} as well as the SoTA CLAS method, and achieve a level of accuracy on par with intra-observer variability.

<p align="center">
<img src="https://i.ibb.co/QjSCnvk/Table1.png" alt="segmentation-results" width=95% /><br>
    <i> Table 1: Average segmentation results at ED and ES. The intra-observer variability results (in red) are taken from the official CAMUS website and are not provided for the left atrium. The best value per column is indicated in bold.</i>
</p>

The LV volumes and EF estimation (see table below) showed that the 2D Dense model outperforms the 2D Sparse model and that the 3D model, in turn, outperforms both. This supports the claim that enforcing temporal consistency in the segmentations is beneficial for EF estimation. Note that the 3D Dense method computes strong EDV and ESV estimates that are well within the intra-observer variability and superior to CLAS. The model's estimation of the EF, however, is slightly less remarkable. Still, we argue that the very low bias and the MAE akin to intra-rater precision advocate sufficiently good estimations of the measure.
<p align="center">
<img src="https://i.ibb.co/LR0P4dK/Table2.png" alt="LV-volume-EF-estimation" width=95% /><br>
    <i> Table 2: LV volume and EF estimation on the test set. The intra-observer variability is indicated in red, and the best column-wise value is displayed in bold.</i>
</p>


The same pattern is observed in the evaluation of the temporal smoothness of the segmentation, as seen in the figure below. The 2D Dense model outperforms the 2D Sparse model and that the 3D model, in turn, outperforms both.
<p align="center">
<img src="https://i.ibb.co/vjp1ZMs/smoothness-evaluation.png" alt="smoothness-evaluation" width=95% /><br>
    <i> Figure 3: Temporal smoothness of the test set prediction (lower values indicate higher smoothness). Note the logarithmic scale on the y-axis.</i>
</p>


Finally, the area curve of a test patient is depicted in the figure below, along with the corresponding ED and ES predictions. The 3D approach appears to be offset from the ground truth and the 2D models, especially at ED and ES. Inspection of other patients revealed that the model is not biased towards over- or under-segmentation of the structures. Rather, the results below suggests the presence of aleatorically uncertain boundaries in the data. There are clear disagreements between the manual and automatic segmentations of the LV myocardium due to the occluded endocardium. An analysis of other test patients unveiled similar discrepancies when the LV myocardium and/or the LA extend beyond the field of view. In these cases, the ambiguous position of the structures is presumably reflected in the creation of the manual annotations. Accordingly, the predictions of our models contain uncertainty, resulting in the observed discrepancy. Future work could model this aleatoric randomness in order to convey the reliability of a given estimation.

<p align="center">
<img src="https://i.ibb.co/s52v9Rh/temporal-consistency-curve-2-4-CH.png" alt="temporal-consistency-curve" width=95% /><br>
    <i> Figure 4: Evaluation of the temporal consistency on patient0002 from the test set. Top row: area curves. Bottom row: corresponding predictions at ED and ES. Following the legend of the curves, the green area refers to the ground truth and the magenta outline is the prediction of the 3D Dense model.</i>
</p>

