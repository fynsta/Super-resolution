import cv2
import torch
import gpytorch
import numpy as np
from enum import Enum

from gp_models import RBFModel, SpectralMixtureModel

class ColorSpace(Enum):
  BGR = 0
  YUV = 1
  GRAYSCALE = 2

SRF = 2
IMAGE_NUMS = [1]

USED_COLOR_SPACE = ColorSpace.YUV # ColorSpace.BGR, ColorSpace.YUV, ColorSpace.GRAYSCALE
USED_MODEL = RBFModel # RBFModel, SpectralMixtureModel
USE_ALL_PIXELS_FOR_TRAINING = False # When False, only samples pixels in a grid pattern
USE_PREDEFINED_HYPERS = True # Only for RBF, SpectralMixtureModel does this automatically
LEARNING_RATE = 0.1
OFFSET_RATE = 0.8 # OFFSET / PATCH_SIZE (0.8 implies around 66% overlap overall)

def get_patch_size(scaling_factor):
  return 20#int(scaling_factor * 10)

def get_offset(scaling_factor):
  # We need to leave a 1px border for the edges
  return int((get_patch_size(scaling_factor)-2) * OFFSET_RATE)

def extract_patch(img, indices, patch_size):
  y, x = indices
  return img[y:y+patch_size, x:x+patch_size].copy()

def extract_neighbors(img, y, x):
  return [img[y_val, x_val] for y_val, x_val in [(y+1, x), (y-1, x), (y, x+1), (y, x-1), (y+1, x+1), (y-1, x-1), (y+1, x-1), (y-1, x+1)]]

# Predicts the pixels in the test image using the training image.
# Note that the test_input is already upscaled, so it has a different size than training_input and training_target.
def predict_pixels(training_input, training_target, test_input, scaling_factor):
  assert training_input.shape == training_target.shape
  assert test_input.shape[0] == training_input.shape[0] * scaling_factor
  assert get_patch_size(scaling_factor) <= training_input.shape[0], f'Patch size {get_patch_size(scaling_factor)} is larger than image size {training_input.shape[0]}'

  upsampled_patch_size = int(get_patch_size(scaling_factor) * scaling_factor)

  patch_indices = []
  for x in range(0, training_input.shape[1] - 1, get_offset(scaling_factor)):
    if x + get_patch_size(scaling_factor) > training_input.shape[1]:
      x = training_input.shape[1] - get_patch_size(scaling_factor)
    for y in range(0, training_input.shape[0] - 1, get_offset(scaling_factor)):
      if y + get_patch_size(scaling_factor) > training_input.shape[0]:
        y = training_input.shape[0] - get_patch_size(scaling_factor)

      if (y, x) not in patch_indices:
        patch_indices.append((y, x))
  upsampled_patch_indices = [(int(y*scaling_factor), int(x*scaling_factor)) for y, x in patch_indices]
  
  if USE_ALL_PIXELS_FOR_TRAINING:
    training_points = [(y, x) for y in range(1, get_patch_size(scaling_factor)-1) for x in range(1, get_patch_size(scaling_factor)-1)]
  else:
    training_indices = [i for i in range(1, get_patch_size(scaling_factor)//2, 2)] + [i for i in range(get_patch_size(scaling_factor)//2, get_patch_size(scaling_factor) - 1, 2)]
    training_points = [(y, x) for y in training_indices for x in training_indices]

  upsampled_patches = []
  for (patch_y, patch_x), (upsampled_patch_y, upsampled_patch_x) in zip(patch_indices, upsampled_patch_indices):
    training_input_patch = extract_patch(training_input, (patch_y, patch_x), get_patch_size(scaling_factor))
    training_target_patch = extract_patch(training_target, (patch_y, patch_x), get_patch_size(scaling_factor))
    train_x = torch.tensor([extract_neighbors(training_input_patch, y, x) for y, x in training_points])
    train_y = torch.tensor([training_target_patch[y, x] for y, x in training_points])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = USED_MODEL(train_x, train_y, likelihood)

    if USE_PREDEFINED_HYPERS and USED_MODEL == RBFModel:
      hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.001),
        'covar_module.base_kernel.lengthscale': torch.tensor(0.223),
        'covar_module.outputscale': train_y.float().var(),
        'mean_module.constant': train_y.float().mean()
      }
      model.initialize(**hypers)

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

    test_input_patch = extract_patch(test_input, (upsampled_patch_y, upsampled_patch_x), upsampled_patch_size)
    test_x = torch.tensor([extract_neighbors(test_input_patch, y, x) for y in range(1, upsampled_patch_size-1) for x in range(1, upsampled_patch_size-1)])

    test_y = likelihood(model(test_x)).mean

    test_input_patch[1:-1, 1:-1] = test_y.reshape(upsampled_patch_size-2, upsampled_patch_size-2).detach().numpy()
    upsampled_patches.append(test_input_patch)

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
  upsampled = np.clip(upsampled, 0, 1)

  return upsampled

def upsample(img, scaling_factor):
  bicubic = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
  prediction = predict_pixels(img, img, bicubic, scaling_factor)

  return prediction

def blur_downsample(img, scaling_factor):
  img = cv2.GaussianBlur(img, (5, 5), 0)
  img = cv2.resize(img, None, fx=1/scaling_factor, fy=1/scaling_factor)
  return img

def deblur(blurred_high, blurred_low, low, scaling_factor):
  return predict_pixels(blurred_low, low, blurred_high, scaling_factor)

def gprsr_for_channel_with_scale(img, scaling_factor):
  blurred_high = upsample(img, scaling_factor)
  blurred_low = blur_downsample(blurred_high, scaling_factor)
  high = deblur(blurred_high, blurred_low, img, scaling_factor)

  return high

def gprsr_for_channel(img):
  img = img.astype(np.double) / 255

  scaling_factor = SRF
  while scaling_factor >= 4:
    img = gprsr_for_channel_with_scale(img, scaling_factor=2)
    scaling_factor /= 2.
  img = gprsr_for_channel_with_scale(img, scaling_factor=scaling_factor)

  img = (img*255).astype(np.uint8)
  return img

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

    
for i in IMAGE_NUMS:
  lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
  lrImg = cv2.imread(lrImagePath)
  
  gprImg = gprsr(lrImg)

  cv2.imwrite(f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR.png', gprImg)
