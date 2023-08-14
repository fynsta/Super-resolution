from enum import Enum
import time
import cv2 as cv
import gpytorch
import numpy as np
from sympy import Ge
import torch
from image_patches import PatchHandler, PatchInterpolationMode
from kernels import GeneralModel, get_kernel_name, matern32_kernel, matern52_kernel, linear_kernel

torch.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)

class ColorSpace(Enum):
  BGR = 0
  YUV = 1
  GRAYSCALE = 2

class Dataset(Enum):
  Set14 = 0
  Set14Smaller = 1 # Created with create_smaller_data.py. Same images as Set14, but 4x smaller.

SRF = 2 # Scaling factor for the overall image (the datasets include images for 2x, 3x and 4x scaling)
DATASET = Dataset.Set14 # The dataset to be used for the algorithm
IMAGE_NUMS = range(1,15) # Image numbers to be used from the used dataset (1-14)
DO_TIMING = True # Whether to print the time it takes to upscale each image

USED_COLOR_SPACE = ColorSpace.GRAYSCALE # Color space to be used for the algorithm
USED_KERNEL = matern52_kernel # supports all kernels from kernels.py
USE_ALL_PIXELS_FOR_TRAINING = True # When False, only samples pixels in a grid pattern
LEARNING_RATE = 0.1 # Learning rate for the hyperparameter training
STRIDE_PERCENTAGE = 0.9 # STRIDE / PATCH_SIZE. A little less than 1 to avoid edge effects.
PATCH_INTERPOLATION = PatchInterpolationMode.AVERAGE # How to interpolate the results of overlapping patches

# Extracts a patch from the image, with the given indices as the top left corner.
# Note that the patch is copied, so changing the patch does not change the image.
def extract_patch(img, y, x, patch_size):
  return img[y:y+patch_size, x:x+patch_size].copy()

# Extracts the 8 neighbors of the pixel at the given indices.
def extract_neighbors(img, y, x):
  return [img[y_val, x_val] for y_val, x_val in [(y+1, x), (y-1, x), (y, x+1), (y, x-1), (y+1, x+1), (y-1, x-1), (y+1, x-1), (y-1, x+1)]]

class GPRSR:
  def __init__(self, scaling_factor, used_kernel, color_mode):
    self.scaling_factor = scaling_factor
    self.current_scaling_factor = None # Used to keep track of the current scaling factor in the algorithm
    self.used_kernel = used_kernel
    self.color_mode = color_mode

  # Apply GPR-SR algorithm for a single image.
  # Handles different color spaces by converting the image to the desired color space before and after the algorithm.
  # Available color spaces are BGR, YUV and grayscale (of course, grayscale will result in a black and white image)
  # Note that the paper uses YIQ, but YUV is very similar in that we only upscale the Y channel (and UV are spanning the same space as IQ)
  def apply(self, lr_image):
    if DO_TIMING:
      start_time = time.time()

    if self.color_mode == ColorSpace.GRAYSCALE:
      img_in_gray = cv.cvtColor(lr_image, cv.COLOR_BGR2GRAY)
      img_out_gray = self.apply_for_channel(img_in_gray)
      img_out = cv.cvtColor(img_out_gray, cv.COLOR_GRAY2BGR)
    elif self.color_mode == ColorSpace.YUV:
      img_in_yuv = cv.cvtColor(lr_image, cv.COLOR_BGR2YUV)
      y, u, v = cv.split(img_in_yuv)
      y = self.apply_for_channel(y)
      u = cv.resize(u, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation=cv.INTER_CUBIC)
      v = cv.resize(v, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation=cv.INTER_CUBIC)
      img_out_yuv = cv.merge((y, u, v))
      img_out = cv.cvtColor(img_out_yuv, cv.COLOR_YUV2BGR)
    else:
      b, g, r = cv.split(lr_image)
      b = self.apply_for_channel(b)
      g = self.apply_for_channel(g)
      r = self.apply_for_channel(r)
      img_out = cv.merge((b, g, r))

    if DO_TIMING:
      print("Upscaling took %s seconds" % (time.time() - start_time), flush=True)

    return img_out
  
  # GPR-SR algorithm for a single channel.
  # Handles higher scaling factors by recursively calling itself with smaller scaling factors.
  # Also converts the data to [0,1] before and after the algorithm to avoid numerical issues.
  def apply_for_channel(self, lr_image_channel):
    # GPR-SR algorithm for a single channel and a single scaling factor.
    # UPSAMPLE -> BLUR AND DOWNSAMPLE -> DEBLUR
    def do_algorithm(img, scaling_factor):
      self.current_scaling_factor = scaling_factor

      blurred_high = self.upsample(img)
      blurred_low = self.blur_downsample(blurred_high)
      high = self.deblur(blurred_high, blurred_low, img)
      
      return high

    img = lr_image_channel.astype(np.double) / 255.

    scaling_factor = self.scaling_factor
    while scaling_factor >= 4:
      scaling_factor /= 2
      img = do_algorithm(img, 2)

    img = do_algorithm(img, scaling_factor)

    return (img * 255).astype(np.uint8)
  
  # The UPSAMPLE step in the paper.
  # Upsamples the image using bicubic interpolation, then tries to predict the pixels 
  # based on data learned from the low resolution image.
  def upsample(self, img):
    bicubic = cv.resize(img, None, fx=self.current_scaling_factor, fy=self.current_scaling_factor, interpolation=cv.INTER_CUBIC)
    prediction = self.predict_pixels(img, img, bicubic)

    return prediction

  # The BLUR AND DOWNSAMPLE step in the paper.
  # Blurs the image using a Gaussian kernel, then downsamples it.
  def blur_downsample(self, img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.resize(img, None, fx=1/self.current_scaling_factor, fy=1/self.current_scaling_factor)
    return img

  # The DEBLUR step in the paper.
  # Tries to predict the pixels in the high resolution image based on the downsampled blurred image.
  # Patches are exactly the same as in the UPSAMPLE step.
  def deblur(self, blurred_high, blurred_low, low):
    return self.predict_pixels(blurred_low, low, blurred_high)
  
  # Predicts the pixels in the test image using the training image.
  # Note that the test_input is already upscaled, so it has a different size than training_input and training_target.
  def predict_pixels(self, training_input, training_target, test_input):
    assert training_input.shape == training_target.shape
    assert test_input.shape[0] == training_input.shape[0] * self.current_scaling_factor

    # According to the paper, the patch size should be around 10x the scaling factor for decent results.
    # (Of course a larger patch size is better, but it also takes longer to train.)
    patch_size = int(self.current_scaling_factor * 10)
    patch_handler = PatchHandler(images = [training_input, training_target, test_input], base_patch_size=patch_size, stride_percentage=STRIDE_PERCENTAGE)

    # Calculate the indices of the training pixels inside the patch.
    # The paper uses a grid pattern to speed up computation, but we can also use all pixels
    if USE_ALL_PIXELS_FOR_TRAINING:
      training_indices = range(1, patch_size-1)
    else:
      training_indices = [i for i in range(1, patch_size//2, 2)] + [i for i in range(patch_size//2, patch_size - 1, 2)]
    
    training_points = [(y, x) for y in training_indices for x in training_indices]

    # The main loop. For each patch, we train a GPR model and predict the pixels in the test image.
    upsampled_patches = []
    for training_input_patch, training_target_patch, test_input_patch in patch_handler:
      # Training data is the 8 neighbors of the training pixels.
      train_x = torch.tensor([extract_neighbors(training_input_patch, y, x) for y, x in training_points])
      # Training labels are the training pixel values themselves.
      train_y = torch.tensor([training_target_patch[y, x] for y, x in training_points])

      # Initialize the model and train it.
      likelihood = gpytorch.likelihoods.GaussianLikelihood()
      model = GeneralModel(self.used_kernel, train_x, train_y, likelihood)

      if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

      hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.001),
        'mean_module.constant': train_y.float().mean()
      }

      if get_kernel_name(self.used_kernel) == 'RBF':
        # These are the hyperparameters advised by the paper (only for the RBF kernel)
        hypers.update({
          'covar_module.base_kernel.lengthscale': torch.tensor(0.223),
          'covar_module.outputscale': train_y.float().var(),
        })
      
      model.initialize(**hypers)

      model.train()
      likelihood.train()
      optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

      for _ in range(50):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

      model.eval()
      likelihood.eval()

      # Use the model to predict the pixels in the test image.
      test_x = torch.tensor([extract_neighbors(test_input_patch, y, x) for y in range(1, test_input_patch.shape[0]-1) for x in range(1, test_input_patch.shape[0]-1)])
      if torch.cuda.is_available():
        test_x = test_x.cuda()
      test_y = likelihood(model(test_x)).mean
      if torch.cuda.is_available():
        test_y = test_y.cpu()

      # This is a bit of a hack. We need to fill in the edges of the image, but we don't want to use the model to predict them.
      # So we just use the original (bicubic) upscaling for the edges and fill the interior with the predicted values.
      test_input_patch[1:-1, 1:-1] = test_y.reshape(test_input_patch.shape[0]-2, test_input_patch.shape[0]-2).detach().numpy()
      upsampled_patches.append(test_input_patch)

    # Rebuild the image from the patches.
    upsampled = patch_handler.combine_patches(upsampled_patches, PATCH_INTERPOLATION,)

    # Clip the values to be between 0 and 1 (valid pixel values), because the model might predict values outside of this range.
    upsampled = np.clip(upsampled, 0, 1)

    return upsampled


# Main loop. Loops over all images in the Set14 dataset and applies the GPR-SR algorithm.
# The paper does not give a dataset, but Set14 is a common benchmark dataset for super resolution.
# Of course, you can also apply the algorithm to your own images.

if __name__ == '__main__':
  gprsr = GPRSR(SRF, USED_KERNEL, USED_COLOR_SPACE)
  kernel_name = get_kernel_name(USED_KERNEL)
  for i in IMAGE_NUMS:
    if DATASET == Dataset.Set14:
      lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
      gprImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR_{kernel_name}{"_gray" if USED_COLOR_SPACE == ColorSpace.GRAYSCALE else ""}.png'
    elif DATASET == Dataset.Set14Smaller:
      lrImagePath = f'Set14_smaller/{i:03d}_LR.png'
      gprImagePath = f'Set14_smaller/{i:03d}_GPR_{kernel_name}_{SRF}x.png'

    lrImg = cv.imread(lrImagePath)
    gprImg = gprsr.apply(lrImg)
    
    cv.imwrite(gprImagePath, gprImg)
