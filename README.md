# DeepMitral
Deep learning based mitral valve segmentation based on 3D Residual UNet from the MONAI framework (https://monai.io/)


Carnahan, P., Moore, J., Bainbridge, D., Eskandari, M., Chen, E.C.S., Peters, T.M. (2021). DeepMitral: Fully Automatic 3D Echocardiography Segmentation for Patient Specific Mitral Valve Modelling. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_44

Trained and tested on MONAI version 0.7 and PyTorch version 1.8.2

### Usage Examples
#### Data Format
Data is expected to be structured using 3 subfolders, train, test, val. Images are expected to having matching names for imaage and correspodning label, with images ending in US and labels ending in label. File format should be .nii. An example of the data structure is here:
```
data
|-- test
|   |-- Patient1-label.nii
|   |-- Patient1-US.nii
|   |-- Patient2-label.nii
|   |-- Patient2-US.nii
|-- train
|   |-- 001-00-label.nii
|   |-- 001-00-US.nii
|   |-- 004-00-label.nii
|   |-- 004-00-US.nii
|-- val
|   |-- 020-00-label.nii
|   |-- 020-00-US.nii
|   |-- 033-00-label.nii
|   |-- 033-00-US.nii
```

Output from both training and validation will be saved in a directory "runs" created in the wokring directory from which DeepMitral is called.

#### Training from epoch 0

```
python deepmitral.py train -data "PATH_TO_DATA"
```
where the data folder corresponds to the top level titled "data" in the example above.

#### Training from a checkpoint

```
python deepmitral.py train -data "PATH_TO_DATA" -load "PATH_TO_CHECKPOINT"
```
where the data folder corresponds to the top level title "data" in the example above and the checkpoint is a .pt file from a prior training run.

#### Running Validation

Validation can be run on either the val images or the test images, with performance metrics being output and model segmentations being saved.

For validation using the val image set:
```
python deepmitral.py validate "PATH_TO_MODEL" -data "PATH_TO_DATA"
```
For validation using the test image set:
```
python deepmitral.py validate "PATH_TO_MODEL" -data "PATH_TO_DATA" -use_test
```
where the data folder corresponds to the top level title "data" in the example above and the model is either a .pt file or a .md file from a prior training run.

#### Running Segmentation

In both training and validation it is expected that images will have corresponding ground truth labels. To use DeepMitral in inference mode on images without ground truth labels, use the segment option. The segment option does not require the directory structure from above, and instead expects to be passed the folder directly containing the target images, and will perform inference on all .nii files. Resulting segmentations will be saved in a subdirectory "out" created in the target image directory.

```
python deepmitral.py segment "PATH_TO_MODEL" "PATH_TO_TARGET_IMAGES"
```
