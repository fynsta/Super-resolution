import cv2
import math
import torch
import gpytorch
import numpy as np

SRF = 2

USE_PREDEFINED_HYPERS = False
USE_YUV = False
USE_ALL_PIXELS_FOR_TRAINING = True
ONLY_PREDICT_CENTER = False
OVERLAP = 0

PATCH_SIZE = SRF * 10
OFFSET = int(PATCH_SIZE * (1 + math.sqrt(1-OVERLAP)) // 2) if OVERLAP > 0 else PATCH_SIZE - 2
UPSAMPLED_PATCH_SIZE = PATCH_SIZE * SRF

print(f'PATCH_SIZE: {PATCH_SIZE}, OFFSET: {OFFSET}')

class ExactGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def extract_patch(img, indices, scale=1):
  y, x = indices
  x *= scale
  y *= scale
  return img[y:y+PATCH_SIZE*scale, x:x+PATCH_SIZE*scale]

def extract_neighbors(img, y, x):
  return [img[y_val, x_val] for y_val, x_val in [(y+1, x), (y-1, x), (y, x+1), (y, x-1), (y+1, x+1), (y-1, x-1), (y+1, x-1), (y-1, x+1)]]

def predict_pixels(training_input, training_target, test_input):
  patch_indices = []
  for x in range(0, training_input.shape[1] - 1, OFFSET):
    if x + PATCH_SIZE > training_input.shape[1]:
      x = training_input.shape[1] - PATCH_SIZE
    for y in range(0, training_input.shape[0] - 1, OFFSET):
      if y + PATCH_SIZE > training_input.shape[0]:
        y = training_input.shape[0] - PATCH_SIZE

      if (y, x) not in patch_indices:
        patch_indices.append((y, x))

  
  if USE_ALL_PIXELS_FOR_TRAINING:
    training_points = [(y, x) for y in range(1, PATCH_SIZE-1) for x in range(1, PATCH_SIZE-1)]
  else:
    # TODO: Cleaner way to implement this sampling?
    training_indices = [i for i in range(1, PATCH_SIZE//2, 2)] + [i for i in range(PATCH_SIZE//2, PATCH_SIZE, 2)]
    training_points = [(y, x) for y in training_indices for x in training_indices]

  upsampled_patches = []
  for patch_y, patch_x in patch_indices:
    training_input_patch = extract_patch(training_input, (patch_y, patch_x))
    training_target_patch = extract_patch(training_target, (patch_y, patch_x))
    if ONLY_PREDICT_CENTER:
      if abs(patch_x - training_input.shape[1]//2) > 50 or abs(patch_y - training_input.shape[0]//2) > 50:
        upsampled_patches.append(cv2.resize(training_input_patch, None, fx=SRF, fy=SRF, interpolation=cv2.INTER_CUBIC))
        continue
    train_x = torch.tensor([extract_neighbors(training_input_patch, y, x) for y, x in training_points])
    train_y = torch.tensor([training_target_patch[y, x] for y, x in training_points])

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    if USE_PREDEFINED_HYPERS:
      hypers = {
        'mean_module.constant': train_y.float().mean(),
        'likelihood.noise_covar.noise': torch.tensor(0.05),
        'covar_module.base_kernel.lengthscale': torch.tensor(0.223),
        'covar_module.outputscale': train_y.float().var()
      }
      model.initialize(**hypers)
    else:
      model.train()
      likelihood.train()
      optimizer = torch.optim.Adam(model.parameters(), lr=10)
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

      for i in range(100):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    test_input_patch = extract_patch(test_input, (patch_y, patch_x), SRF)
    test_x = torch.tensor([extract_neighbors(test_input_patch, y, x) for y in range(1, UPSAMPLED_PATCH_SIZE-1) for x in range(1, UPSAMPLED_PATCH_SIZE-1)])

    test_y = model(test_x).mean
    test_input_patch[1:-1, 1:-1] = test_y.reshape(UPSAMPLED_PATCH_SIZE-2, UPSAMPLED_PATCH_SIZE-2).detach().numpy()
    upsampled_patches.append(test_input_patch)

  upscaled = np.zeros(test_input.shape)
  overlapped_counter = np.zeros(test_input.shape)
  for y, x, upsampled_patch in zip(*patch_indices, upsampled_patches):
    x *= SRF
    y *= SRF
    
    x_start = 0 if x == 0 else 1
    y_start = 0 if y == 0 else 1
    x_end = UPSAMPLED_PATCH_SIZE if x + UPSAMPLED_PATCH_SIZE == upscaled.shape[1] else UPSAMPLED_PATCH_SIZE - 1
    y_end = UPSAMPLED_PATCH_SIZE if y + UPSAMPLED_PATCH_SIZE == upscaled.shape[0] else UPSAMPLED_PATCH_SIZE - 1

    for j in range(y_start, y_end):
      for k in range(x_start, x_end):
        upscaled[y+j, x+k] += upsampled_patch[j, k]
        overlapped_counter[y+j, x+k] += 1

  upscaled /= overlapped_counter
  upscaled.clip(0, 255)

  return upscaled.astype(np.uint8)

def upsample(img):
  bicubic = cv2.resize(img, None, fx=SRF, fy=SRF, interpolation=cv2.INTER_CUBIC)

  return predict_pixels(img, img, bicubic)


def blur_downsample(img):
  # TODO: Use pyrDown instead?
  img = cv2.GaussianBlur(img, (5, 5), 0)
  img = cv2.resize(img, None, fx=1/SRF, fy=1/SRF, interpolation=cv2.INTER_CUBIC)
  return img

def deblur(blurred_high, blurred_low, low):
  return predict_pixels(blurred_low, low, blurred_high)

def gprsr_for_channel(img):
  blurred_high = upsample(img)
  blurred_low = blur_downsample(blurred_high)
  high = deblur(blurred_high, blurred_low, img)
  return high

def gprsr(img):
  if USE_YUV:
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

    
for i in range(1, 2):
  lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
  hrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_HR.png'
  lrImg = cv2.imread(lrImagePath)
  hrImg = cv2.imread(hrImagePath)
  
  gprImg = gprsr(lrImg)

  cv2.imwrite(f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR.png', gprImg)

  # psnr = torchmetrics.functional.peak_signal_noise_ratio(torch.tensor(imgOutBgr), torch.tensor(hrImg))
  # ssim = torchmetrics.functional.structural_similarity_index_measure(torch.tensor(imgOutBgr), torch.tensor(hrImg))
  # print(f'PSNR: {psnr}, SSIM: {ssim}')
