# Gaussian Processes for Super Resolution
This repository contains the code for implementation of the approach outlined in the paper [Single image super-resolution using Gaussian process regression](https://ieeexplore.ieee.org/document/5995713) by He & Siu. As the name suggests, the approach uses Gaussian Process Regression to perform super-resolution on a single image.

The code is written in Python 3.10.12 and uses the [GPytorch](https://gpytorch.ai//) library for Gaussian Process Regression. The code is tested on MacOS 13.4.1.

Test pictures are taken from the Set14 dataset, which was downloaded from [here](https://github.com/jbhuang0604/SelfExSR). These can be found in the `Set14` folder. I have also generated smaller versions of the images, which can be found in the `Set14_smaller` folder.

Currently, the following kernels are implemented and tested:

- RBF (Squared Exponential)
- Matern 3/2
- Matern 5/2
- Exponential

In general, the Matern 3/2 and Matern 5/2 kernels perform better than the other options.

## Usage

The main entry point for the code is the `main.py` file. The file contains the code for generating the high resolution image from the low resolution image. There are many options for user customization, which I have tried to document in the file itself.

Additionally, two helpful shell scripts for analyzing the results (i.e. calulating SSIM and PSNR values) are provided. Note that these scripts require [ImageMagick](https://imagemagick.org/script/download.php) to be installed. The scripts are:

- `comparison_methods.sh`: This script calculates the average SSIM and PSNR for each of the methods (bicubic interpolation, different kernels in Gaussian Process Regression, etc.).
- `comparison_images.sh`: This script calculates the SSIM and PSNR for each of the images. This allows us to see which images are better suited for the different methods.
