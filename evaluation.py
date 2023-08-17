from collections import defaultdict
from enum import Enum
from typing import NamedTuple
import cv2
import os
from gpytorch.kernels import Kernel
from skimage import metrics
from sklearn import preprocessing
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
    if self.number_of_images == 0:
      return PerceptualSimilarity(0,0,0)
    return PerceptualSimilarity(self.psnr / self.number_of_images, self.ssim / self.number_of_images, 1)


class Evaluator:
  def __init__(self, srf, image_nums = range(1,15), verbose=False, generate_gpr_images=False):
    self.srf = srf
    self.image_nums = image_nums
    self.verbose = verbose
    self.metrics = defaultdict(lambda: PerceptualSimilarity(0,0,0))
    self.generate_gpr_images = generate_gpr_images

  def clear(self):
    self.metrics = defaultdict(lambda: PerceptualSimilarity(0,0,0))

  def get_hr_image(self, i):
    return self.get_image(i, 'HR', fail_on_nonexistent=True)
  
  def get_bicubic_image(self, i):
    return self.get_image(i, 'bicubic', fail_on_nonexistent=True)
  
  def get_agpr_image(self, i):
    return self.get_image(i, 'AGPR')
  
  def get_gpr_image(self, i, kernel):
    kernel_name = get_kernel_name(kernel)
    image = self.get_image(i, f'GPR_{kernel_name}')
    if image is None and self.generate_gpr_images:
      lr_image = cv2.imread(f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_LR.png')
      image = GPRSR(self.srf, kernel, USED_COLOR_SPACE).apply(lr_image)
      cv2.imwrite(f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_GPR_{kernel_name}.png', image)
    return image

  def get_image(self, i, method, fail_on_nonexistent=False):
    image_path = f'Set14/image_SRF_{self.srf}/img_{i:03d}_SRF_{self.srf}_{method}.png'
    if not(os.path.exists(image_path)):
      if fail_on_nonexistent:
        raise Exception(f'Image {image_path} does not exist')
      else:
        return None

    return cv2.imread(image_path)
  
  def get_metrics(self, hr_image, upsampled_image, name):
    if upsampled_image is None:
      return PerceptualSimilarity(0,0,0)
    
    preprocessed_hr_image = self.preprocess_image(hr_image)
    preprocessed_upsampled_image = self.preprocess_image(upsampled_image)

    psnr = self.get_psnr(preprocessed_hr_image, preprocessed_upsampled_image)
    ssim = self.get_ssim(preprocessed_hr_image, preprocessed_upsampled_image)
    if self.verbose:
      print(f'{name} - PSNR: {psnr:.3f}, SSIM: {ssim:.3f}')
    return PerceptualSimilarity(psnr, ssim)
  
  def get_psnr(self, hr_image, upsampled_image, preprocess=False):
    if preprocess:
      hr_image = self.preprocess_image(hr_image)
      upsampled_image = self.preprocess_image(upsampled_image)
    return metrics.peak_signal_noise_ratio(hr_image, upsampled_image, data_range=255)
  
  def get_ssim(self, hr_image, upsampled_image, preprocess=False):
    if preprocess:
      hr_image = self.preprocess_image(hr_image)
      upsampled_image = self.preprocess_image(upsampled_image)
    return metrics.structural_similarity(hr_image, upsampled_image, data_range=255)
  
  def preprocess_image(self, image):
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[self.srf:-self.srf, self.srf:-self.srf]
  
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

  def evaluate_method(self, method, metric = PerceptualSimilarityMetric.ALL):
    kernel_name = get_kernel_name(method) if isinstance(method, Kernel) else method

    if self.verbose:
      print(f'Method {kernel_name}')

    result = PerceptualSimilarity(0,0,0)
    for i in self.image_nums:
      hr_image = self.get_hr_image(i)
      upsampled_image = self.get_gpr_image(i, method) if isinstance(method, Kernel) else self.get_image(i, method)
      result += self.get_metrics(hr_image, upsampled_image, kernel_name)
    
    result = result.average()
    if self.verbose:
      print(f'Average - PSNR: {result.psnr:.3f}, SSIM: {result.ssim:.3f}', flush=True)

    if metric == PerceptualSimilarityMetric.PSNR:
      return result.psnr
    elif metric == PerceptualSimilarityMetric.SSIM:
      return result.ssim

  def evaluate(self):
    for i in self.image_nums:
      self.evaluate_image(i)

    for key, value in self.metrics.items():
      self.metrics[key] = value.average()

    print('Averages')
    for key, value in self.metrics.items():
      print(f'{key} - PSNR: {value.psnr:.3f}, SSIM: {value.ssim:.3f}')
    
    return self.metrics

if __name__ == '__main__':
  evaluator = Evaluator(2, verbose=True)
  evaluator.evaluate_method('AGPR', PerceptualSimilarityMetric.ALL)