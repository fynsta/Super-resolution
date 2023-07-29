import cv2
import torch
import gpytorch
import numpy as np
from enum import Enum

from gp_models import ExponentialModel, Matern32Model, Matern52Model, RBFModel, SpectralMixtureModel

class ColorSpace(Enum):
  BGR = 0
  YUV = 1
  GRAYSCALE = 2

class Dataset(Enum):
  Set14 = 0
  Set14Smaller = 1 # Created with create_smaller_data.py. Same images as Set14, but 4x smaller.

# Scaling factor for the overall image (the datasets include images for 2x, 3x and 4x scaling)
SRF = 2
# The dataset to be used for the algorithm
DATASET = Dataset.Set14Smaller
# Image numbers to be used from the used dataset (1-14)
IMAGE_NUMS = range(1, 15)

USED_COLOR_SPACE = ColorSpace.YUV
# TODO: Add support for SpectralMixtureModel
USED_MODEL = Matern32Model # currently supports RBFModel, ExponentialModel, MaternModel32, MaternModel52
USE_ALL_PIXELS_FOR_TRAINING = True # When False, only samples pixels in a grid pattern
USE_PREDEFINED_HYPERS = False # Only for RBF (SpectralMixtureModel does this inherently)
LEARNING_RATE = 0.1 # Learning rate for the hyperparameter training
STRIDE_PERCENT = 0.8 # STRIDE / PATCH_SIZE

# According to the paper, the patch size should be around 10x the scaling factor for decent results.
# (Of course a larger patch size is better, but it also takes longer to train.)
def get_patch_size(scaling_factor):
  return int(scaling_factor * 10)

# The stride is the distance between two patches, both horizontally and vertically.
# Note that we need to subtract 2 from the patch size to account for the edges (which are not predicted).
# The paper uses an overlap of 66% overall (which means a stride rate of around 0.8 according to my rough calculations)
def get_stride(scaling_factor):
  return int((get_patch_size(scaling_factor)-2) * STRIDE_PERCENT)

# Extracts a patch from the image, with the given indices as the top left corner.
# Note that the patch is copied, so changing the patch does not change the image.
def extract_patch(img, indices, patch_size):
  y, x = indices
  return img[y:y+patch_size, x:x+patch_size].copy()

# Extracts the 8 neighbors of the pixel at the given indices.
def extract_neighbors(img, y, x):
  return [img[y_val, x_val] for y_val, x_val in [(y+1, x), (y-1, x), (y, x+1), (y, x-1), (y+1, x+1), (y-1, x-1), (y+1, x-1), (y-1, x+1)]]

# Predicts the pixels in the test image using the training image.
# Note that the test_input is already upscaled, so it has a different size than training_input and training_target.
def predict_pixels(training_input, training_target, test_input, scaling_factor):
  assert training_input.shape == training_target.shape
  assert test_input.shape[0] == training_input.shape[0] * scaling_factor
  assert get_patch_size(scaling_factor) <= training_input.shape[0], f'Patch size {get_patch_size(scaling_factor)} is larger than image size {training_input.shape[0]}'

  upsampled_patch_size = int(get_patch_size(scaling_factor) * scaling_factor)

  # Calculating the indices of the patches to be used for training.
  # Patches that would go out of bounds are not used are moved to the interior so that they are within bounds.
  patch_indices = []
  for x in range(0, training_input.shape[1] - 1, get_stride(scaling_factor)):
    if x + get_patch_size(scaling_factor) > training_input.shape[1]:
      x = training_input.shape[1] - get_patch_size(scaling_factor)
    for y in range(0, training_input.shape[0] - 1, get_stride(scaling_factor)):
      if y + get_patch_size(scaling_factor) > training_input.shape[0]:
        y = training_input.shape[0] - get_patch_size(scaling_factor)

      patch_indices.append((y, x))

  # Obviously, the upsampled patch indices are just the patch indices multiplied by the scaling factor.
  upsampled_patch_indices = [(int(y*scaling_factor), int(x*scaling_factor)) for y, x in patch_indices]
  
  # Calculate the indices of the training pixels inside the patch.
  # The paper uses a grid pattern to speed up computation, but we can also use all pixels
  if USE_ALL_PIXELS_FOR_TRAINING:
    training_indices = range(1, get_patch_size(scaling_factor)-1)
  else:
    training_indices = [i for i in range(1, get_patch_size(scaling_factor)//2, 2)] + [i for i in range(get_patch_size(scaling_factor)//2, get_patch_size(scaling_factor) - 1, 2)]
  
  training_points = [(y, x) for y in training_indices for x in training_indices]

  # The main loop. For each patch, we train a model and predict the pixels in the test image.
  upsampled_patches = []
  for (patch_y, patch_x), (upsampled_patch_y, upsampled_patch_x) in zip(patch_indices, upsampled_patch_indices):
    # Extract the training data from the patch.
    training_input_patch = extract_patch(training_input, (patch_y, patch_x), get_patch_size(scaling_factor))
    training_target_patch = extract_patch(training_target, (patch_y, patch_x), get_patch_size(scaling_factor))
    train_x = torch.tensor([extract_neighbors(training_input_patch, y, x) for y, x in training_points])
    train_y = torch.tensor([training_target_patch[y, x] for y, x in training_points])

    # Initialize the model and train it.
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = USED_MODEL(train_x, train_y, likelihood)

    if USE_PREDEFINED_HYPERS and USED_MODEL == RBFModel:
      # These are the hyperparameters from the paper for the RBF model.
      # I'm not sure if they are really optimal, maybe they did something a little different with the data.
      # And obviously, the optimal hyperparameters depend on the data itself.
      hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.001),
        'covar_module.base_kernel.lengthscale': torch.tensor(0.223),
        'covar_module.outputscale': train_y.float().var(),
        'mean_module.constant': train_y.float().mean()
      }
      model.initialize(**hypers)

    # TODO: We currently do a hyperparameter tuning loop for each patch, but we could also do this for only a few patches and then use the model for all patches.
    # This would easily speed up the algorithm by a high factor, but might result in worse predictions.
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(100):
      optimizer.zero_grad()
      output = model(train_x)
      loss = -mll(output, train_y)
      loss.backward()
      optimizer.step()

    model.eval()
    likelihood.eval()

    # Use the model to predict the pixels in the test image.
    test_input_patch = extract_patch(test_input, (upsampled_patch_y, upsampled_patch_x), upsampled_patch_size)
    test_x = torch.tensor([extract_neighbors(test_input_patch, y, x) for y in range(1, upsampled_patch_size-1) for x in range(1, upsampled_patch_size-1)])
    test_y = likelihood(model(test_x)).mean

    # This is a bit of a hack. We need to fill in the edges of the image, but we don't want to use the model to predict them.
    # So we just use the original (bicubic) upscaling for the edges and fill the interior with the predicted values.
    test_input_patch[1:-1, 1:-1] = test_y.reshape(upsampled_patch_size-2, upsampled_patch_size-2).detach().numpy()
    upsampled_patches.append(test_input_patch)


  # Now we need to combine the patches into a single image.
  # Overlapping patches are averaged together.
  upsampled = np.zeros(test_input.shape)
  overlapped_counter = np.zeros(test_input.shape)
  for (y, x), upsampled_patch in zip(upsampled_patch_indices, upsampled_patches):    
    # Considerations for edges. Since they are not predicted by the model, they should not be used.
    # Except at the edge of the image, where they are used to fill in the missing pixels.
    # These values are straight from the original (bicubic) upscaling.
    x_start = 0 if x == 0 else 1
    y_start = 0 if y == 0 else 1
    x_end = upsampled_patch_size if x + upsampled_patch_size == upsampled.shape[1] else upsampled_patch_size - 1
    y_end = upsampled_patch_size if y + upsampled_patch_size == upsampled.shape[0] else upsampled_patch_size - 1

    for j in range(y_start, y_end):
      for k in range(x_start, x_end):
        upsampled[y+j, x+k] += upsampled_patch[j, k]
        overlapped_counter[y+j, x+k] += 1

  upsampled /= overlapped_counter

  # Clip the values to be between 0 and 1 (valid pixel values), because the model might predict values outside of this range.
  upsampled = np.clip(upsampled, 0, 1)

  return upsampled

# The UPSAMPLE step in the paper.
# Upsamples the image using bicubic interpolation, then uses the model to predict the pixels 
# based on data learned from the low resolution image.
def upsample(img, scaling_factor):
  bicubic = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
  prediction = predict_pixels(img, img, bicubic, scaling_factor)

  return prediction

# The BLUR AND DOWNSAMPLE step in the paper.
# Blurs the image using a Gaussian kernel, then downsamples it.
def blur_downsample(img, scaling_factor):
  img = cv2.GaussianBlur(img, (5, 5), 0)
  img = cv2.resize(img, None, fx=1/scaling_factor, fy=1/scaling_factor)
  return img

# The DEBLUR step in the paper.
# Uses the model to predict the pixels in the high resolution image based on the downsampled blurred image.
# Patches are exactly the same as in the UPSAMPLE step.
def deblur(blurred_high, blurred_low, low, scaling_factor):
  return predict_pixels(blurred_low, low, blurred_high, scaling_factor)

# GPR-SR algorithm for a single channel and a single scaling factor.
# UPSAMPLE -> BLUR AND DOWNSAMPLE -> DEBLUR
def gprsr_for_channel_with_scale(img, scaling_factor):
  blurred_high = upsample(img, scaling_factor)
  blurred_low = blur_downsample(blurred_high, scaling_factor)
  high = deblur(blurred_high, blurred_low, img, scaling_factor)

  return high

# GPR-SR algorithm for a single channel.
# Handles higher scaling factors by recursively calling itself with smaller scaling factors.
# Also converts the data to [0,1] before and after the algorithm to avoid numerical issues.
def gprsr_for_channel(img):
  img = img.astype(np.double) / 255

  scaling_factor = SRF
  while scaling_factor >= 4:
    img = gprsr_for_channel_with_scale(img, scaling_factor=2)
    scaling_factor /= 2.
  img = gprsr_for_channel_with_scale(img, scaling_factor=scaling_factor)

  img = (img*255).astype(np.uint8)
  return img

# GPR-SR algorithm for a single image.
# Handles different color spaces by converting the image to the desired color space before and after the algorithm.
# Available color spaces are BGR, YUV and grayscale (of course, grayscale will result in a black and white image)
# Note that the paper uses YIQ, but YUV is very similar in that we only upscale the Y channel (and UV are spanning the same space as IQ)
def gprsr(img):
  if USED_COLOR_SPACE == ColorSpace.GRAYSCALE:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgOut = gprsr_for_channel(img)
  elif USED_COLOR_SPACE == ColorSpace.YUV:
    imgInYuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(imgInYuv)
    u = cv2.resize(u, None, fx=SRF, fy=SRF, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, None, fx=SRF, fy=SRF, interpolation=cv2.INTER_CUBIC)

    y = gprsr_for_channel(y)
    imgOutYuv = cv2.merge((y, u, v))
    imgOut = cv2.cvtColor(imgOutYuv, cv2.COLOR_YUV2BGR)
  else:
    b, g, r = cv2.split(img)
    b = gprsr_for_channel(b)
    g = gprsr_for_channel(g)
    r = gprsr_for_channel(r)
    imgOut = cv2.merge((b, g, r))

  return imgOut

# Main loop. Loops over all images in the Set14 dataset and applies the GPR-SR algorithm.
# The paper does not give a dataset, but Set14 is a common benchmark dataset for super resolution.
# Of course, you can also apply the algorithm to your own images.
for model in [RBFModel, ExponentialModel, Matern32Model, Matern52Model]:
  USED_MODEL = model
  print(f'Using model {USED_MODEL._get_name()}')
  for i in IMAGE_NUMS:
    print(f'Processing image {i}')
    if DATASET == Dataset.Set14:
      lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
      gprImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR_{USED_MODEL._get_name()}.png'
    elif DATASET == Dataset.Set14Smaller:
      lrImagePath = f'Set14_smaller/{i:03d}_LR.png'
      gprImagePath = f'Set14_smaller/{i:03d}_GPR_{USED_MODEL._get_name()}_{SRF}x.png'

    lrImg = cv2.imread(lrImagePath)
    
    gprImg = gprsr(lrImg)
    cv2.imwrite(gprImagePath, gprImg)
