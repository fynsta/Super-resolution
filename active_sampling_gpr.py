import gpytorch
from math import floor
import torch
import cv2 as cv
import numpy as np
import os
from enum import Enum
from datasets import load_dataset

from evaluation import Evaluator, PerceptualSimilarityMetric
from kernels import GeneralModel, linear_kernel

SRF = 2

N = 500 # Number of available patches to train on in the images
R = 100 # Number of patches to actually train on (chosen by active sampling)
K = 10 # Number of nearest neighbors to use for characteristic score
L = 7 # Size of the patches
assert L % 2 == 1, "L must be odd"
padding = floor(L / 2)

class TrainingDataset(Enum):
  Set14 = 1
  BSD100 = 2

DATASET = TrainingDataset.Set14

SCALE_COEFFICIENT_BANDWIDTH = 1
CHARACTERISTIC_TRADEOFF = 0.2
TAU = 0.01 # Iterative back projection parameter

class AGPRSuperResolution:
  def __init__(self, super_resolution_factor, use_existing_model = False) -> None:
    self.super_resolution_factor = super_resolution_factor

    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if not(os.path.exists(self.get_checkpoint_folder())):
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
    bicubic_images = [cv.resize(
      lr_image, 
      (lr_image.shape[1] * self.super_resolution_factor, lr_image.shape[0] * self.super_resolution_factor), 
      interpolation=cv.INTER_CUBIC
    ) for lr_image in lr_images]
    hr_images = self.get_images("HR")

    high_frequency_images = [hr_image - bicubic_image for hr_image, bicubic_image in zip(hr_images, bicubic_images)]

    # Choose random patches as basis for the dataset
    self.original_dataset = []
    for _ in range(N):
      image_index = np.random.randint(0, len(hr_images))
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
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255
    
    upscaled_image = cv.resize(image, (image.shape[1] * self.super_resolution_factor, image.shape[0] * self.super_resolution_factor), interpolation=cv.INTER_CUBIC)

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

    predictions = predictions.numpy().reshape((upscaled_image.shape[0] - self.super_resolution_factor * padding, upscaled_image.shape[1] - self.super_resolution_factor * padding))
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
      blurred = cv.GaussianBlur(current_image, (3, 3), 0)
      downscaled = cv.resize(blurred, (original_image.shape[1], original_image.shape[0]), interpolation=cv.INTER_CUBIC)
      difference = original_image - downscaled
      if current_error - 0.01 <= np.linalg.norm(difference)**2: break
      current_error = np.linalg.norm(difference)**2

      upscaled_difference = cv.resize(difference, (current_image.shape[1], current_image.shape[0]), interpolation=cv.INTER_CUBIC)
      back_projection = cv.filter2D(upscaled_difference, -1, back_projection_kernel)
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
    if DATASET == TrainingDataset.Set14:
      image_paths = [f"Set14/image_SRF_{self.super_resolution_factor}/img_{image_number:03d}_SRF_{self.super_resolution_factor}_{resolution}.png" for image_number in range(1,15)]
    elif DATASET == TrainingDataset.BSD100:
      image_paths = load_dataset('eugenesiow/BSD100', split='validation')[resolution.lower()]
    images = [cv.imread(image_path) for image_path in image_paths]
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255 for image in images]
    return images
  
  def get_checkpoint_path(self, filename):
    return os.path.join(self.get_checkpoint_folder(), filename + ".pt")
  
  def get_checkpoint_folder(self):
    return f"checkpoints/{DATASET.name}/SRF_{self.super_resolution_factor}/"

if __name__ == "__main__":
  sparse_gpr = AGPRSuperResolution(SRF, use_existing_model=True)

  for i in range(2,3):
    lr_image = cv.imread(f"Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png")
    interpolated_image = sparse_gpr.upscale(lr_image)
    cv.imwrite(f"Set14/image_SRF_{SRF}/img_{i:03d}_SRF_2_AGPR.png", interpolated_image)

  evaluation = Evaluator(SRF, range(1,15), verbose=True)
  evaluation.evaluate(PerceptualSimilarityMetric.SSIM)

  ## Evaluation code (hacked together)
  # image_num = 1
  # agpr_interpolation = sparse_gpr.upscale(cv.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_LR.png"))
  # cv.imwrite(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_AGPR.png", agpr_interpolation)

  # # Compare SSIM with bicubic interpolation & GPR
  # hr_image = cv.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_HR.png")
  # hr_image_gray = cv.cvtColor(hr_image, cv.COLOR_BGR2GRAY)
  # lr_image = cv.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_LR.png")
  # lr_image_gray = cv.cvtColor(lr_image, cv.COLOR_BGR2GRAY)
  # gpr_image = cv.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_GPR_matern52_gray.png", cv.IMREAD_GRAYSCALE)
  # agpr_image = cv.imread(f"Set14/image_SRF_2/img_{image_num:03d}_SRF_2_AGPR.png", cv.IMREAD_GRAYSCALE)
  # bicubic_interpolation = cv.resize(lr_image_gray, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv.INTER_CUBIC)
  # print("SSIM bicubic interpolation:", metrics.structural_similarity(hr_image_gray, bicubic_interpolation))
  # print("SSIM SRGPR interpolation:", metrics.structural_similarity(hr_image_gray, gpr_image))
  # print("SSIM AGPR interpolation:", metrics.structural_similarity(hr_image_gray, agpr_image))
