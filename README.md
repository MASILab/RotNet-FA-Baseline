# RotNet-FA-Image

## Overview
RotNet-FA-Image is a Python-based deep learning project for predicting rotation angles of fractional anisotropy (FA) maps derived from NIfTI (.nii.gz) files. It leverages 3D convolutional neural networks (CNNs) implemented in PyTorch to regress rotation angles. The augmented data helps us extract meaningful features from the input data.

Original RotNet repo: https://github.com/d4nst/RotNet

## Environment
- Ubuntu version: 22.04.4 LTS 
- CUDA version: 12.2
- GPU: NVIDIA RTX A4000

## Installation
```
# Create and activate a new conda environment
conda create -n isbi_challenge python=3.10 -y
conda activate isbi_challenge

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip3 install -r requirements.txt
```

## Preprocessing Script (preproc_fa.sh)
- Assumes [PreQual](https://github.com/MASILab/PreQual/) has already been run on input data -- we recommend that all challenge participants do this, as the testing data will be preprocessed this way
- Converts the .mha/.json data available from grand-challenge.org to .nii.gz/.bval/.bvec format that the preprocessing steps need
- Skull stripping
- Tensor fitting
- FA map generation

These steps may or may not be necessary based on your approach.
Though PreQual outputs include FA maps, your Docker needs include any steps beyond preprocessing so that the training/testing data can be generated in the same way.

## Project Structure
- `NiiDataset`: Custom PyTorch Dataset class for loading and preprocessing NIfTI files.
- `RotationRegressionModel`: 3D CNN for predicting rotation angles.
- `Training Script`: Handles the training process, model evaluation, and visualization.

## Running the Model
- Prepare your dataset of .nii.gz files and place them in the data directory. (default: `./data`)
- Train the model using the provided training script:
```python
python3 rotnet3D_regression.py
```
- The model's training progress will be logged, and visualization outputs will be saved in the result_regression directory.

## Outputs
- Useful Feature: Saved as .json file in the format needed for evaluation code.
- Training History: Plots of the mean squared error (MSE) loss over epochs.
<img src="result_regression/training_history.png" alt="sample" width="500" height="400">

- Visualizations: Image without rotation and image with ground truth rotation / Ground truth rotation and the predicted rotation for epoch 1, 25, 50, 75, 100
<img src="result_regression/epoch_1/sample_prediction_0.png" alt="Epoch 1" width="900" height="600">
<img src="result_regression/epoch_25/sample_prediction_0.png" alt="Epoch 25" width="900" height="600">
<img src="result_regression/epoch_50/sample_prediction_0.png" alt="Epoch 50" width="900" height="600">
<img src="result_regression/epoch_75/sample_prediction_0.png" alt="Epoch 75" width="900" height="600">
<img src="result_regression/epoch_100/sample_prediction_0.png" alt="Epoch 100" width="900" height="600">
