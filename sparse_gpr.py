import gpytorch
from math import floor
import torch
import cv2
import numpy as np
from kernels import GeneralModel, linear_kernel, matern52_kernel
from skimage import metrics

M = 1 # Number of images to train on
N = 500 # Number of available patches to train on in the images
R = 100 # Number of patches to actually train on (chosen by active sampling)
K = 10 # Number of nearest neighbors to use for characteristic score
L = 7 # Size of the patches
assert L % 2 == 1, "L must be odd"
padding = floor(L / 2)

SCALE_COEFFICIENT_BANDWIDTH = 1
CHARACTERISTIC_TRADEOFF = 0.2

class SparseGPRSuperResolution:
  def __init__(self, super_resolution_factor) -> None:
    self.super_resolution_factor = super_resolution_factor

    self.train()

  def train(self) -> None:
    lr_images = [cv2.imread(get_image_path(self.super_resolution_factor, i+1, "LR")) for i in range(M)]
    lr_images = [cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY) / 255 for lr_image in lr_images]
    bicubic_images = [cv2.resize(
      lr_image, 
      (lr_image.shape[1] * self.super_resolution_factor, lr_image.shape[0] * self.super_resolution_factor), 
      interpolation=cv2.INTER_CUBIC
    ) for lr_image in lr_images]
    hr_images = [cv2.imread(get_image_path(self.super_resolution_factor, i+1, f"HR")) for i in range(M)]
    hr_images = [cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY) / 255 for hr_image in hr_images]

    high_frequency_images = [hr_image - bicubic_image for hr_image, bicubic_image in zip(hr_images, bicubic_images)]

    # Choose random patches
    self.original_dataset = []
    coords = []
    for i in range(N):
      image_index = np.random.randint(0, M)
      hr_image, high_frequency_image = hr_images[image_index], high_frequency_images[image_index]
      x, y = np.random.randint(padding, hr_image.shape[1] - padding), np.random.randint(padding, hr_image.shape[0] - padding)
      hr_patch = hr_image[y-padding:y+padding+1, x-padding:x+padding+1].reshape(-1)
      center_pixel = high_frequency_image[y, x]
      self.original_dataset.append(np.concatenate((hr_patch, [center_pixel])))
      coords.append((image_index, x, y))

    self.original_dataset, retained_indices = np.unique(np.array(self.original_dataset), axis=0, return_index=True)
    coords = [coords[i] for i in retained_indices]

    # Create a picture where chosen patches are surrounded by red border
  
    # Choose patches to train on using active sampling
    distances = [[(np.linalg.norm(datapoint - other_datapoint))**2 for other_datapoint in self.original_dataset if (datapoint != other_datapoint).any()] for datapoint in self.original_dataset]
    bandwidth = SCALE_COEFFICIENT_BANDWIDTH * np.median(distances)
    self.available_dataset = np.copy(self.original_dataset)
    self.dataset = []
    for i in range(R):
      characteristic_scores = [self.get_characteristic_score(datapoint, bandwidth) for datapoint in self.available_dataset]
      max_characteristic_score_index = np.argmax(characteristic_scores)
      max_characteristic_score_datapoint = self.available_dataset[max_characteristic_score_index]
      self.available_dataset = np.delete(self.available_dataset, max_characteristic_score_index, axis=0)
      self.dataset.append(max_characteristic_score_datapoint)

      _,x,y = coords.pop(max_characteristic_score_index)
      cv2.rectangle(hr_image, (x-padding, y-padding), (x+padding, y+padding), (0, 0, 255), 1)
    cv2.imwrite("hr_image_with_boxes.png", hr_image * 255)

    # Train the model
    self.train_x = torch.tensor(np.array([datapoint[:-1] for datapoint in self.dataset]))
    self.train_y = torch.tensor([datapoint[-1] for datapoint in self.dataset])

    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    self.model = GeneralModel(linear_kernel, self.train_x, self.train_y, self.likelihood)
    self.model.parameters

    if torch.cuda.is_available():
      self.train_x = self.train_x.cuda()
      self.train_y = self.train_y.cuda()
      self.model = self.model.cuda()
      self.likelihood = self.likelihood.cuda()

    self.model.train()
    self.likelihood.train()

    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    for _ in range(50):
      optimizer.zero_grad()
      output = self.model(self.train_x)
      loss = -mll(output, self.train_y)
      loss.backward()
      optimizer.step()

    self.model.eval()
    self.likelihood.eval()

  def upscale(self, image : np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
    image = cv2.resize(image, (image.shape[1] * self.super_resolution_factor, image.shape[0] * self.super_resolution_factor), interpolation=cv2.INTER_CUBIC)

    interpolated_image = image.copy()

    patches = []
    for i in range(padding, image.shape[0] - padding):
      for j in range(padding, image.shape[1] - padding):
        patches.append(image[i-padding:i+padding+1, j-padding:j+padding+1].reshape(-1))

    test_x = torch.tensor(np.array(patches)).float()
    if torch.cuda.is_available():
      test_x = test_x.cuda()

    predictions = self.model(test_x).mean.detach()
    if torch.cuda.is_available():
      predictions = predictions.cpu()

    predictions = predictions.numpy().reshape((image.shape[0] - 2 * padding, image.shape[1] - 2 * padding))
    interpolated_image[padding:-padding, padding:-padding] += predictions

    return interpolated_image * 255

  def get_characteristic_score(self, datapoint, bandwidth) -> float:
    representativeness = self.get_representativeness(datapoint, bandwidth)
    diversity = self.get_diversity(datapoint, bandwidth)

    return CHARACTERISTIC_TRADEOFF * representativeness + (1 - CHARACTERISTIC_TRADEOFF) * diversity

  def get_representativeness(self, datapoint, bandwidth) -> float:
    distances = np.array([np.linalg.norm(datapoint - other_datapoint)**2 for other_datapoint in self.original_dataset if (datapoint != other_datapoint).any() ])
    neighbor_distances = np.sort(distances)[:K]

    return np.mean(np.exp(-neighbor_distances / (2*bandwidth)))
  
  def get_diversity(self, datapoint, bandwidth) -> float:
    selected_distances = np.array([np.linalg.norm(datapoint - other_datapoint)**2 for other_datapoint in self.dataset ])

    if len(selected_distances) == 0: return 0

    return np.min(-np.exp(-selected_distances / (2*bandwidth)))

def get_image_path(super_resolution_factor, image_number, resolution):
  return f"Set14/image_SRF_{super_resolution_factor}/img_{image_number:03d}_SRF_{super_resolution_factor}_{resolution}.png"
  
# if __name__ == "__main__":
#   sparse_gpr = SparseGPRSuperResolution(2)
#   interpolation = sparse_gpr.upscale(cv2.imread("Set14/image_SRF_2/img_000_SRF_2_LR.png"))
#   cv2.imwrite("interpolation.png", interpolation)

# Compare SSIM with bicubic interpolation
hr_image = cv2.imread("Set14/image_SRF_2/img_000_SRF_2_HR.png")
hr_image_gray = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
lr_image = cv2.imread("Set14/image_SRF_2/img_000_SRF_2_LR.png")
lr_image_gray = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)
bicubic_interpolation = cv2.resize(lr_image_gray, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
print("SSIM bicubic interpolation:", metrics.structural_similarity(hr_image_gray, bicubic_interpolation))
spgpr_interpolation = cv2.imread("interpolation.png")
spgpr_interpolation_gray = cv2.cvtColor(spgpr_interpolation, cv2.COLOR_BGR2GRAY)
print("SSIM spgpr interpolation:", metrics.structural_similarity(hr_image_gray, spgpr_interpolation_gray))
