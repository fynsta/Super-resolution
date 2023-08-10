import os
import cv2 as cv
import numpy as np
import torch
from evaluation import Evaluator, PerceptualSimilarityMetric
from main import SRF, USE_ALL_PIXELS_FOR_TRAINING, USED_COLOR_SPACE, IMAGE_NUMS, GPRSR, extract_neighbors, extract_patch
from kernels import GeneralModel, rbf_kernel, matern52_kernel, matern32_kernel, exponential_kernel, periodic_kernel, spectral_mixture_kernel, linear_kernel

from gpytorch.kernels import AdditiveKernel, RBFKernel, MaternKernel, PeriodicKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood

### Automatic model finder according to Duvenaud's paper
# Still WIP because I have not implemented the accuracy function yet

class AutomaticModelConstructor():
  # base_kernels: list of kernels to be used as base kernels
  # train_x: training data
  # train_y: training labels
  # likelihood: likelihood function
  # verbose: whether to print out the best kernel
  # interpretability_mode: whether to use interpretability mode (i.e. only multiply the kernel onto the last kernel in the sum)
  # num_iterations: number of iterations to run the algorithm
  def __init__(self, base_kernels, train_x, train_y, likelihood, verbose=False, interpretability_mode=True, num_iterations=5):
    self.verbose = verbose
    self.train_x = train_x
    self.train_y = train_y
    self.likelihood = likelihood
    self.interpretability_mode = interpretability_mode

    self.evaluator = Evaluator(SRF)

    self.base_kernels = base_kernels
    self.num_iterations = num_iterations
  
  def run(self):
    self.find_best_start_model()
    while self.num_iterations > 0:
      if not self.construct_next_model():
        break
      self.num_iterations -= 1
    
  def find_best_start_model(self):
    self.current_best_ssim = 0
    self.current_best_kernel = None
    
    for kernel in self.base_kernels:
      ssim = self.get_accuracy(kernel)

      if ssim > self.current_best_ssim:
        self.current_best_ssim = ssim
        self.current_best_kernel = kernel

    if self.verbose:
      print(f'Best kernel: {self.current_best_kernel}')

  def construct_next_model(self):
    next_best_ssim = 0
    next_best_kernel = None

    # Adding a kernel
    for k in self.base_kernels:
      kernel = self.current_best_kernel + k
      ssim = self.get_accuracy(kernel)

      if ssim > next_best_ssim:
        next_best_ssim = ssim
        next_best_kernel = kernel

    # Multiplying a kernel
    for k in self.base_kernels:
      if self.interpretability_mode and self.current_best_kernel.__class__ == AdditiveKernel:
        kernel = self.current_best_kernel
        kernel.kernels[-1] = kernel.kernels[-1] * k
      else:
        kernel = self.current_best_kernel * k

      ssim = self.get_accuracy(kernel)

      if ssim > next_best_ssim:
        next_best_ssim = ssim
        next_best_kernel = kernel

    if next_best_ssim > self.current_best_ssim:
      self.current_best_ssim = next_best_ssim
      self.current_best_kernel = next_best_kernel
    else:
      return False

    if self.verbose:
      print(f'Best kernel: {next_best_kernel}')
    
    return True

  def get_accuracy(self, kernel):
    model = self.get_model(kernel)
    
    return np.random.random()
    # TODO: What to use for accuracy? SSIM of the generated images? Or the marginal likelihood of the model?

    # gprsr = GPRSR(SRF, model, USED_COLOR_SPACE)

    # for i in IMAGE_NUMS:
    #   lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
    #   gprImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR_{get_kernel_name(kernel)}.png'
    #   if os.path.exists(gprImagePath):
    #     continue

    #   lrImg = cv.imread(lrImagePath)
    #   gprImg = gprsr.apply(lrImg)

    #   cv.imwrite(gprImagePath, gprImg)

    # ssim = self.evaluator.evaluate_metric(model, PerceptualSimilarityMetric.SSIM)
    # if self.verbose:
    #   print(f'{get_kernel_name(kernel)} - SSIM: {ssim:.3f}')

    # return ssim

  def get_model(self, kernel):
    return GeneralModel(kernel, self.train_x, self.train_y, self.likelihood)

if __name__ == '__main__':
  image_path = f'Set14/image_SRF_{SRF}/img_001_SRF_{SRF}_LR.png'
  image = cv.imread(image_path)
  upscaledImage = cv.resize(image, None, fx=SRF, fy=SRF, interpolation=cv.INTER_CUBIC)
  patch_size = 20
  patch = extract_patch(upscaledImage, 0, 0, patch_size=patch_size)

  if USE_ALL_PIXELS_FOR_TRAINING:
    training_indices = range(1, patch_size-1)
  else:
    training_indices = [i for i in range(1, patch_size//2, 2)] + [i for i in range(patch_size//2, patch_size - 1, 2)]
  
  training_points = [(y, x) for y in training_indices for x in training_indices]
  train_x = torch.tensor(np.array([extract_neighbors(patch, y, x) for y, x in training_points]))
  train_y = torch.tensor(np.array([patch[y, x] for y, x in training_points]))
  
  likelihood = GaussianLikelihood()
  m = AutomaticModelConstructor([
    rbf_kernel,
    matern52_kernel,
    matern32_kernel,
    # exponential_kernel,
    periodic_kernel,
    # spectral_mixture_kernel,
    linear_kernel
  ], train_x, train_y, likelihood, verbose=True)
  m.run()