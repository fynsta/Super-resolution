from collections import defaultdict
from enum import Enum
from typing import NamedTuple
import cv2
import os
from skimage import metrics
from kernels import get_kernel_name, rbf_kernel, exponential_kernel, matern32_kernel, matern52_kernel, spectral_mixture_kernel, linear_kernel
from main import GPRSR, USED_COLOR_SPACE

class PerceptualSimilarityMetric(Enum):
  PSNR = 1
  SSIM = 2
  ALL = 3

class PerceptualSimilarity(NamedTuple):
  psnr: float
  ssim: float
  number_of_images: int = 1

  def __add__(self, other):
    return PerceptualSimilarity(self.psnr + other.psnr, self.ssim + other.ssim, self.number_of_images + other.number_of_images)
  
  def average(self):
    return PerceptualSimilarity(self.psnr / self.number_of_images, self.ssim / self.number_of_images, 1)


class Evaluator:
  def __init__(self, srf, image_nums = range(1,15), verbose=False):
    self.srf = srf
    self.image_nums = image_nums
    self.verbose = verbose
    self.metrics = defaultdict(lambda: PerceptualSimilarity(0,0,0))

  def clear(self):
    self.metrics = defaultdict(lambda: PerceptualSimilarity(0,0,0))

  def get_hr_image(self, i):
    return self.get_image(i, 'HR')
  
  def get_bicubic_image(self, i):
    return self.get_image(i, 'bicubic')
  
  def get_agpr_image(self, i):
    return self.get_image(i, 'AGPR')
  
  def get_gpr_image(self, i, kernel):
    kernel_name = get_kernel_name(kernel)
    image = self.get_image(i, f'GPR_{kernel_name}', fail_on_nonexistent=False)
    if image is None:
      lr_image = cv2.imread(f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_LR.png')
      image = GPRSR(self.srf, kernel, USED_COLOR_SPACE).apply(lr_image)
      cv2.imwrite(f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_GPR_{kernel_name}.png', image)

      # Read the image again to make sure it is in the correct format for the evaluator
      image = self.get_image(i, f'GPR_{kernel_name}')
    return image

  def get_image(self, i, method, fail_on_nonexistent=True, grayscale=True):
    image_path = f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_{method}.png'
    if not(os.path.exists(image_path)):
      if fail_on_nonexistent:
        raise Exception(f'Image {image_path} does not exist')
      else:
        return None

    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[self.srf:-self.srf, self.srf:-self.srf]
  
  def get_metrics(self, hr_image, upsampled_image, name):
    psnr = metrics.peak_signal_noise_ratio(hr_image, upsampled_image)
    ssim = metrics.structural_similarity(hr_image, upsampled_image)
    if self.verbose:
      print(f'{name} - PSNR: {psnr:.3f}, SSIM: {ssim:.3f}')
    return PerceptualSimilarity(psnr, ssim)
  
  def evaluate_image(self, i):
    if self.verbose:
      print(f'Image {i}')

    hr_image = self.get_hr_image(i)
    bicubic_image = self.get_bicubic_image(i)
    self.metrics['Bicubic (Baseline)'] += self.get_metrics(hr_image, bicubic_image, 'Bicubic (Baseline)')

    agpr_image = self.get_agpr_image(i)
    self.metrics['AGPR'] += self.get_metrics(hr_image, agpr_image, 'AGPR')

    for kernel in [rbf_kernel, exponential_kernel, matern32_kernel, matern52_kernel, spectral_mixture_kernel, linear_kernel]:
      kernel_name = get_kernel_name(kernel)
      gpr_image = self.get_gpr_image(i, kernel)
      if gpr_image is not None:
        self.metrics[kernel_name] += self.get_metrics(hr_image, gpr_image, kernel_name)

  def evaluate_metric(self, kernel, metric):
    kernel_name = get_kernel_name(kernel)

    result = PerceptualSimilarity(0,0,0)
    for i in self.image_nums:
      hr_image = self.get_hr_image(i)
      upsampled_image = self.get_gpr_image(i, kernel)
      result += self.get_metrics(hr_image, upsampled_image, kernel_name)
    
    result = result.average()
    if self.verbose:
      print(f'{kernel_name} - PSNR: {result.psnr:.3f}, SSIM: {result.ssim:.3f}', flush=True)

    if metric == PerceptualSimilarityMetric.PSNR:
      return result.psnr
    elif metric == PerceptualSimilarityMetric.SSIM:
      return result.ssim

  def evaluate(self):
    for i in self.image_nums:
      self.evaluate_image(i)

    if self.verbose:
      print('Averages')
      for key, value in self.metrics.items():
        print(f'{key} - PSNR: {value.psnr/14:.3f}, SSIM: {value.ssim/14:.3f}')
    
    return self.metrics
  
if __name__ == '__main__':
  evaluator = Evaluator(2, verbose=True)
  ssim = evaluator.evaluate_metric(rbf_kernel + rbf_kernel, PerceptualSimilarityMetric.SSIM)
  print(ssim)