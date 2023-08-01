import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from itertools import product

# Two-dimensional training data sampled from a sinc function
train_x = torch.rand(1000, 2)
train_y = torch.sin(train_x[:, 0] * (2 * math.pi)) + torch.sin(train_x[:, 1] * (2 * math.pi)) + torch.randn(train_x.size(0)) * 0.2
print(train_x.shape)
print(train_y.shape)

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2, log_mixture_weights=torch.randn(10), log_mixture_means=torch.randn(10, 2), log_mixture_scales=torch.randn(10, 2))
  
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
  
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):
  optimizer.zero_grad()
  output = model(train_x)
  loss = -mll(output, train_y)
  loss.backward()
  print('Iter %d/%d - Loss: %.3f' % (i + 1, 50, loss.item()))
  optimizer.step()

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
  test_x = torch.tensor([(x,y) for x in torch.linspace(0, 5, 10) for y in torch.linspace(0, 5, 10)])
  print(test_x.shape)
  observed_pred = likelihood(model(test_x))
  print(observed_pred.mean)

