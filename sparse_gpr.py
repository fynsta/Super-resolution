import gpytorch
from math import floor
import torch
import cv2
import numpy as np
from kernels import GeneralModel, linear_kernel
from skimage import metrics
import os

M = 13 # Number of images to train on
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
  def __init__(self, super_resolution_factor, use_existing_model = False) -> None:
    self.super_resolution_factor = super_resolution_factor
    self.image_numbers = [i for i in range(1, M+1)]

    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if not(os.path.exists(self.get_checkpoint_path("model"))):
      use_existing_model = False

    if use_existing_model:
      self.train_x = torch.load(self.get_checkpoint_path("train_x"))
      self.train_y = torch.load(self.get_checkpoint_path("train_y"))
    else:
      self.set_training_data()

    if torch.cuda.is_available():
      self.train_x = self.train_x.cuda()
      self.train_y = self.train_y.cuda()
      self.model = self.model.cuda()
      self.likelihood = self.likelihood.cuda()

    self.model = GeneralModel(linear_kernel, self.train_x, self.train_y, self.likelihood)

    if use_existing_model:
      self.model.load_state_dict(torch.load(self.get_checkpoint_path("model")))
    else:
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

      torch.save(self.model.state_dict(), self.get_checkpoint_path("model"))

    self.model.eval()
    self.likelihood.eval()

  def set_training_data(self) -> None:
    lr_images = self.get_images("LR")
    bicubic_images = [cv2.resize(
      lr_image, 
      (lr_image.shape[1] * self.super_resolution_factor, lr_image.shape[0] * self.super_resolution_factor), 
      interpolation=cv2.INTER_CUBIC
    ) for lr_image in lr_images]
    hr_images = self.get_images("HR")

    high_frequency_images = [hr_image - bicubic_image for hr_image, bicubic_image in zip(hr_images, bicubic_images)]

    # Choose random patches as basis for the dataset
    self.original_dataset = []
    for _ in range(N):
      image_index = np.random.randint(0, M)
      hr_image, high_frequency_image = hr_images[image_index], high_frequency_images[image_index]
      x, y = np.random.randint(padding, hr_image.shape[1] - padding), np.random.randint(padding, hr_image.shape[0] - padding)
      hr_patch = hr_image[y-padding:y+padding+1, x-padding:x+padding+1].reshape(-1)
      center_pixel = high_frequency_image[y, x]
      self.original_dataset.append(np.concatenate((hr_patch, [center_pixel])))

    self.original_dataset = np.unique(np.array(self.original_dataset), axis=0)
  
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

    torch.save(self.train_x, self.get_checkpoint_path("train_x"))
    torch.save(self.train_y, self.get_checkpoint_path("train_y"))

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

    back_projected_image = self.iterative_back_projection(interpolated_image, image)

    return back_projected_image * 255
  
  def iterative_back_projection(self, interpolated_image : np.ndarray, original_image : np.ndarray) -> np.ndarray:
    max_iterations = 100
    back_projection_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    current_image = interpolated_image.copy()
    current_error = np.inf

    for _ in range(max_iterations):
      blurred = cv2.GaussianBlur(current_image, (3, 3), 0)
      downscaled = cv2.resize(blurred, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
      difference = original_image - downscaled
      if current_error - 0.01 <= np.linalg.norm(difference)**2: break
      current_error = np.linalg.norm(difference)**2

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
  
  def get_images(self, resolution):
    return [self.get_image(self.super_resolution_factor, resolution, i) for i in self.image_numbers]

  def get_image(self, resolution, image_number):
    image_path = f"Set14/image_SRF_{self.super_resolution_factor}/img_{image_number:03d}_SRF_{self.super_resolution_factor}_{resolution}.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
    return image
  
  def get_checkpoint_path(self, filename):
    return f"checkpoints/SRF_{self.super_resolution_factor}/{'-'.join([str(image_number) for image_number in self.image_numbers])}/{filename}.pth"
  

if __name__ == "__main__":
  sparse_gpr = SparseGPRSuperResolution(2)

  ## Evaluation code (hacked together)
  image_num = 14
  spgpr_interpolation = sparse_gpr.upscale(cv2.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_LR.png"))
  cv2.imwrite(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_SPGPR.png", spgpr_interpolation)

  # Compare SSIM with bicubic interpolation & GPR
  hr_image = cv2.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_HR.png")
  hr_image_gray = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
  lr_image = cv2.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_LR.png")
  lr_image_gray = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)
  gpr_image = cv2.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_GPR_matern52_gray.png", cv2.IMREAD_GRAYSCALE)
  spgpr_image = cv2.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_SPGPR.png", cv2.IMREAD_GRAYSCALE)
  bicubic_interpolation = cv2.resize(lr_image_gray, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
  print("SSIM bicubic interpolation:", metrics.structural_similarity(hr_image_gray, bicubic_interpolation))
  print("SSIM standard gpr interpolation:", metrics.structural_similarity(hr_image_gray, gpr_image))
  print("SSIM spgpr interpolation:", metrics.structural_similarity(hr_image_gray, spgpr_image))
