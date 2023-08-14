from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel, SpectralMixtureKernel, AdditiveKernel, ScaleKernel, LinearKernel, ProductKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from sklearn import base

rbf_kernel = ScaleKernel(RBFKernel())
matern52_kernel = ScaleKernel(MaternKernel(nu=2.5))
matern32_kernel = ScaleKernel(MaternKernel(nu=1.5))
exponential_kernel = ScaleKernel(MaternKernel(nu=0.5))
periodic_kernel = ScaleKernel(PeriodicKernel())
spectral_mixture_kernel = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=8)
linear_kernel = ScaleKernel(LinearKernel())

class GeneralModel(ExactGP):
  def __init__(self, covar_module, train_x, train_y, likelihood):
    super(GeneralModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = ConstantMean()
    self.covar_module = covar_module

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return MultivariateNormal(mean_x, covar_x)
  
def get_kernel_name(kernel):
  if isinstance(kernel, ScaleKernel):
    return get_kernel_name(kernel.base_kernel)
  elif isinstance(kernel, RBFKernel):
    return "RBF"
  elif isinstance(kernel, MaternKernel):
    nu = "52" if kernel.nu == 2.5 else "32" if kernel.nu == 1.5 else "12" if kernel.nu == 0.5 else str(kernel.nu)
    return f"matern{nu}"
  elif isinstance(kernel, PeriodicKernel):
    return "periodic"
  elif isinstance(kernel, SpectralMixtureKernel):
    return "spectral_mixture"
  elif isinstance(kernel, LinearKernel):
    return "linear"
  elif isinstance(kernel, AdditiveKernel):
    subkernel_names = [get_kernel_name(k) for k in kernel.kernels]
    return f"add({','.join(subkernel_names)})"
  elif isinstance(kernel, ProductKernel):
    subkernel_names = [get_kernel_name(k) for k in kernel.kernels]
    return f"prod({','.join(subkernel_names)})"
  else:
    raise Exception(f"Unknown kernel type: {kernel}")