import gpytorch
from math import floor
import torch
import cv2
import numpy as np
from kernels import GeneralModel, linear_kernel, matern52_kernel
from skimage import metrics, transform

M = 14 # Number of images to train on
N = 500 # Number of available patches to train on in the images
R = 100 # Number of patches to actually train on (chosen by active sampling)
K = 10 # Number of nearest neighbors to use for characteristic score
L = 7 # Size of the patches
assert L % 2 == 1, "L must be odd"
padding = floor(L / 2)

SCALE_COEFFICIENT_BANDWIDTH = 1
CHARACTERISTIC_TRADEOFF = 0.2
TAU = 0.01 # Iterative back projection parameter

class SparseGPRSuperResolution:
  def __init__(self, super_resolution_factor) -> None:
    self.super_resolution_factor = super_resolution_factor

    self.train()

  def train(self) -> None:
    lr_images = get_images(self.super_resolution_factor, "LR")
    bicubic_images = [cv2.resize(
      lr_image, 
      (lr_image.shape[1] * self.super_resolution_factor, lr_image.shape[0] * self.super_resolution_factor), 
      interpolation=cv2.INTER_CUBIC
    ) for lr_image in lr_images]
    hr_images = get_images(self.super_resolution_factor, "HR")

    high_frequency_images = [hr_image - bicubic_image for hr_image, bicubic_image in zip(hr_images, bicubic_images)]

    # Choose random patches
    self.original_dataset = []
    for _ in range(N):
      image_index = np.random.randint(0, M)
      hr_image, high_frequency_image = hr_images[image_index], high_frequency_images[image_index]
      x, y = np.random.randint(padding, hr_image.shape[1] - padding), np.random.randint(padding, hr_image.shape[0] - padding)
      hr_patch = hr_image[y-padding:y+padding+1, x-padding:x+padding+1].reshape(-1)
      center_pixel = high_frequency_image[y, x]
      self.original_dataset.append(np.concatenate((hr_patch, [center_pixel])))

    self.original_dataset = np.unique(np.array(self.original_dataset), axis=0)

    # Create a picture where chosen patches are surrounded by red border
  
    # Choose patches to train on using active sampling
    distances = [[(np.linalg.norm(datapoint - other_datapoint))**2 for other_datapoint in self.original_dataset if (datapoint != other_datapoint).any()] for datapoint in self.original_dataset]
    bandwidth = SCALE_COEFFICIENT_BANDWIDTH * np.median(distances)
    self.available_dataset = np.copy(self.original_dataset)
    self.dataset = []
    for _ in range(R):
      characteristic_scores = [self.get_characteristic_score(datapoint, bandwidth) for datapoint in self.available_dataset]
      max_characteristic_score_index = np.argmax(characteristic_scores)
      max_characteristic_score_datapoint = self.available_dataset[max_characteristic_score_index]
      self.available_dataset = np.delete(self.available_dataset, max_characteristic_score_index, axis=0)
      self.dataset.append(max_characteristic_score_datapoint)

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
    
    upscaled_image = cv2.resize(image, (image.shape[1] * self.super_resolution_factor, image.shape[0] * self.super_resolution_factor), interpolation=cv2.INTER_CUBIC)

    interpolated_image = upscaled_image.copy()

    patches = []
    for i in range(padding, upscaled_image.shape[0] - padding):
      for j in range(padding, upscaled_image.shape[1] - padding):
        patches.append(upscaled_image[i-padding:i+padding+1, j-padding:j+padding+1].reshape(-1))

    test_x = torch.tensor(np.array(patches)).float()
    if torch.cuda.is_available():
      test_x = test_x.cuda()

    predictions = self.model(test_x).mean.detach()
    if torch.cuda.is_available():
      predictions = predictions.cpu()

    predictions = predictions.numpy().reshape((upscaled_image.shape[0] - 2 * padding, upscaled_image.shape[1] - 2 * padding))
    interpolated_image[padding:-padding, padding:-padding] += predictions
    interpolated_image = np.clip(interpolated_image, 0, 1)
    interpolated_image *= 255

    back_projected_image = interpolated_image.copy()
    #back_projected_image = self.iterative_back_projection(interpolated_image, image)

    return back_projected_image
  
  def iterative_back_projection(self, interpolated_image : np.ndarray, original_image) -> np.ndarray:
    max_iterations = 50
    back_projection_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    hr_image = cv2.imread("Set14/image_SRF_2/img_000_SRF_2_HR.png")
    hr_image_gray = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)

    print('Starting iterative back projection')

    current_image = interpolated_image.copy()

    for _ in range(max_iterations):
      blurred = cv2.GaussianBlur(current_image, (3, 3), 0)
      downscaled = cv2.resize(blurred, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
      difference = original_image - downscaled
      print('Error: ', np.linalg.norm(difference)**2)
      print('SSIM: ', metrics.structural_similarity(hr_image_gray, current_image))
      upscaled_difference = cv2.resize(difference, (current_image.shape[1], current_image.shape[0]), interpolation=cv2.INTER_CUBIC)
      back_projection = cv2.filter2D(upscaled_difference, -1, back_projection_kernel)
      current_image += back_projection * TAU

    return current_image

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
  
def get_images(super_resolution_factor, resolution):
  return [get_image(super_resolution_factor, resolution, i) for i in range(1, M+1)]

def get_image(super_resolution_factor, resolution, image_number):
  image_path = f"Set14/image_SRF_{super_resolution_factor}/img_{image_number:03d}_SRF_{super_resolution_factor}_{resolution}.png"
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
  return image
  
if __name__ == "__main__":
  sparse_gpr = SparseGPRSuperResolution(2)
  # original_image = get_image(2, "HR", 1)
  # bicubic_interpolation = cv2.resize(original_image, (original_image.shape[1] // 2, original_image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
  # cv2.imwrite("bicubic_interpolation.png", bicubic_interpolation * 255)
  # result = sparse_gpr.iterative_back_projection(bicubic_interpolation, original_image)
  # cv2.imwrite("result.png", result * 255)

  interpolation = sparse_gpr.upscale(cv2.imread("Set14/image_SRF_2/img_000_SRF_2_LR.png"))
  cv2.imwrite("interpolation.png", interpolation)

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
