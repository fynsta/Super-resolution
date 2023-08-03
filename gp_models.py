from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel, SpectralMixtureKernel, AdditiveKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

rbf_kernel = ScaleKernel(RBFKernel())
matern52_kernel = ScaleKernel(MaternKernel(nu=2.5))
matern32_kernel = ScaleKernel(MaternKernel(nu=1.5))
exponential_kernel = ScaleKernel(MaternKernel(nu=0.5))
periodic_kernel = ScaleKernel(PeriodicKernel())
spectral_mixture_kernel = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=8)

class GeneralModel(ExactGP):
  def __init__(self, train_x, train_y, likelihood, covar_module):
    super(GeneralModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = ConstantMean()
    self.covar_module = covar_module

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return MultivariateNormal(mean_x, covar_x)
  
  @staticmethod
  def _get_name():
    return "base"

class RBFModel(GeneralModel):
  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood, rbf_kernel)

  def _get_name():
    return "RBF"

class Matern52Model(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    super().__init__(train_inputs, train_targets, likelihood, matern52_kernel)

  def _get_name():
    return "matern52"

class Matern32Model(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    super().__init__(train_inputs, train_targets, likelihood, matern32_kernel)

  def _get_name():
    return "matern32"
  
class ExponentialModel(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    super().__init__(train_inputs, train_targets, likelihood, exponential_kernel)

  def _get_name():
    return "exponential"
  
class PeriodicModel(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    super().__init__(train_inputs, train_targets, likelihood, periodic_kernel)

  def _get_name():
    return "periodic"

class SpectralMixtureModel(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=8)
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "spectral_mixture"
  
class MaternWithPeriodic(GeneralModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = AdditiveKernel(ScaleKernel(MaternKernel(nu=1.5)), ScaleKernel(PeriodicKernel()))
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "matern_with_periodic"