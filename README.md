# Cat vs. Dog Classification Enhanced by SRGAN

## Introduction
This repository is dedicated to the task of binary classification of cats and dogs using a Super-Resolution Generative Adversarial Network (SRGAN) to enhance image quality. The SRGAN approach is based on the paper by Ledig et al., which can be found [here](https://arxiv.org/abs/1609.04802).

The goal of this project is to compare the performance of a binary classification model trained on standard resolution images (128x128) against the same model trained on images that have been downscaled and then upscaled using SRGAN (32x32 to 128x128).

## Repository Structure

/
├── checkpoints/ #First model checkpoints and TensorBoard logs.
├── cat_vs_dog_dataset/ # Original dataset for the classification task.
├── srgan/ # SRGAN implementation and related files.
│ ├── train.py # Script for training the SRGAN.
│ ├── dataset.py # Dataset handling for the SRGAN.
│ ├── model.py # SRGAN model architecture.
│ └── utils.py # Utilities for SRGAN training and image processing.
├── train.py # Script for training the binary classification model.
├── test.py # Script for testing the binary classification model.
└── README.md # Documentation for this repository.


## Getting Started

To use this repository for cat and dog classification:

1. Clone the repository to your local machine.
2. Download the Kaggle dataset from [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and place it in the `cat_vs_dog_dataset` directory.
3. Run `train.py` to train the binary classification model on the original high-resolution images.
4. Execute the scripts within the `srgan` directory to train the SRGAN model and upscale the low-resolution images.
5. Use the enhanced images to retrain the binary classification model and compare the results with the original model.

## Results

The results section should detail the performance metrics of the binary classification model before and after the application of SRGAN, highlighting the impact of image resolution on model accuracy.

## Contributions

Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## Acknowledgements
- Credit to the authors of the SRGAN paper for their foundational work in image super-resolution.
- Thanks to Kaggle for providing the dataset for this classification challenge.

## Contact
For any queries or support, please open an issue in the repository or contact the repository maintainers directly.
