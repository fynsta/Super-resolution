

from main import SRF
from evaluation import Evaluator, PerceptualSimilarityMetric
import copy
from kernels import GeneralModel, get_kernel_name, rbf_kernel, matern52_kernel, matern32_kernel, exponential_kernel, periodic_kernel, spectral_mixture_kernel, linear_kernel

from gpytorch.kernels import AdditiveKernel

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
  def __init__(self, base_kernels, image_nums = range(1,15), verbose=False, interpretability_mode=True, num_iterations=5):
    self.verbose = verbose
    self.interpretability_mode = interpretability_mode

    self.evaluator = Evaluator(SRF, image_nums=image_nums)

    self.base_kernels = base_kernels
    self.num_iterations = num_iterations
  
  def run(self):
    self.find_best_start_model()
    while self.num_iterations > 0:
      if not self.construct_next_model():
        break
      self.num_iterations -= 1
    return self.current_best_kernel
    
  def find_best_start_model(self):
    self.current_best_ssim = 0
    self.current_best_kernel = None
    
    for kernel in self.base_kernels:
      ssim = self.get_ssim(kernel)

      if ssim > self.current_best_ssim:
        self.current_best_ssim = ssim
        self.current_best_kernel = kernel

    if self.verbose:
      print(f'Best kernel: {self.current_best_kernel}', flush=True)

  def construct_next_model(self):
    next_best_ssim = 0
    next_best_kernel = None

    # Adding a kernel
    for k in self.base_kernels:
      kernel = copy.deepcopy(self.current_best_kernel)
      kernel = kernel + k
      ssim = self.get_ssim(kernel)

      if ssim > next_best_ssim:
        next_best_ssim = ssim
        next_best_kernel = kernel

    # Multiplying a kernel
    for k in self.base_kernels:
      kernel = copy.deepcopy(self.current_best_kernel)
      if self.interpretability_mode and kernel.__class__ == AdditiveKernel:
        kernel.kernels[-1] = kernel.kernels[-1] * k
      else:
        kernel = self.current_best_kernel * k

      ssim = self.get_ssim(kernel)

      if ssim > next_best_ssim:
        next_best_ssim = ssim
        next_best_kernel = kernel

    if next_best_ssim > self.current_best_ssim:
      self.current_best_ssim = next_best_ssim
      self.current_best_kernel = next_best_kernel
    else:
      return False

    if self.verbose:
      print(f'Best kernel: {next_best_kernel}', flush=True)
    
    return True

  def get_ssim(self, kernel):
    try:
      ssim = self.evaluator.evaluate_method(kernel, PerceptualSimilarityMetric.SSIM)
      if self.verbose:
        print(f'SSIM for {get_kernel_name(kernel)}: {ssim}', flush=True)
      return ssim
    except:
      return 0

if __name__ == '__main__':
  best_kernel = AutomaticModelConstructor(
    [rbf_kernel, matern52_kernel, exponential_kernel, periodic_kernel, linear_kernel],
    image_nums=[1],
    verbose=True
  ).run()
  print('Finally done!')
  print(f'Best kernel: {best_kernel}')