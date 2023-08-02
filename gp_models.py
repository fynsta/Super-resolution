import gpytorch

class BaseModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, covar_module):
    super(BaseModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = covar_module

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
  
  @staticmethod
  def _get_name():
    raise NotImplementedError

class RBFModel(BaseModel):
  def __init__(self, train_x, train_y, likelihood):
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    super().__init__(train_x, train_y, likelihood, covar_module)

  def _get_name():
    return "RBF"

class Matern52Model(BaseModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "matern52"

class Matern32Model(BaseModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "matern32"
  
class ExponentialModel(BaseModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "exponential"
  
class PeriodicModel(BaseModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "periodic"

class SpectralMixtureModel(BaseModel):
  def __init__(self, train_inputs, train_targets, likelihood):
    covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=8)
    #covar_module.initialize_from_data(train_inputs, train_targets)
    super().__init__(train_inputs, train_targets, likelihood, covar_module)

  def _get_name():
    return "spectral_mixture"