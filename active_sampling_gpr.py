import gpytorch
from math import floor
import torch
import cv2 as cv
import numpy as np
import os
from enum import Enum
from time import time
from datasets import load_dataset

from evaluation import Evaluator
from kernels import GeneralModel, linear_kernel

torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)

SRF = 3

N = 10000 # Number of available patches to train on in the images
ACTIVE_SAMPLING_RATIO = 0.1 # Ratio of patches to actually train on
K = 10 # Number of nearest neighbors to use for characteristic score
L = 7 # Size of the patches
assert L % 2 == 1, "L must be odd"
padding = floor(L / 2)

class TrainingDataset(Enum):
  Set14 = 1
  BSD100 = 2

class ColorSpace(Enum):
  YUV = 1
  GRAYSCALE = 2

DATASET = TrainingDataset.BSD100
COLOR_SPACE = ColorSpace.GRAYSCALE

SCALE_COEFFICIENT_BANDWIDTH = 0.2
CHARACTERISTIC_TRADEOFF = 0.2
TAU = 0.01 # Iterative back projection parameter

class AGPRSuperResolution:
  def __init__(self, srf, use_existing_model = False, verbose = False) -> None:
    self.srf = srf
    self.verbose = verbose
    self.start_time = time()

    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if not(os.path.exists(self.get_checkpoint_path('model'))):
      use_existing_model = False

    if use_existing_model:
      self.train_x = torch.load(self.get_checkpoint_path("train_x"))
      self.train_y = torch.load(self.get_checkpoint_path("train_y"))
    else:
      self.set_training_data()

    self.model = GeneralModel(linear_kernel, self.train_x, self.train_y, self.likelihood)

    if torch.cuda.is_available():
      self.train_x = self.train_x.cuda()
      self.train_y = self.train_y.cuda()
      self.model = self.model.cuda()
      self.likelihood = self.likelihood.cuda()

    if use_existing_model:
      self.model.load_state_dict(torch.load(self.get_checkpoint_path("model")))
    else:
      self.model.train()
      self.likelihood.train()

      if self.verbose: print("Training model...", flush=True)
      training_start_time = time()

      optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

      for _ in range(100):
        optimizer.zero_grad()
        output = self.model(self.train_x)
        loss = -mll(output, self.train_y)
        loss.backward()
        optimizer.step()

      torch.save(self.model.state_dict(), self.get_checkpoint_path("model"))

      if self.verbose: print("Model trained in {:.2f} seconds".format(time() - training_start_time))

    self.model.eval()
    self.likelihood.eval()

  def set_training_data(self) -> None:
    training_data_time = time()

    hr_images = self.get_training_images("HR")
    hr_images = [hr_image if hr_image.shape[0] % self.srf == 0 else hr_image[:-(hr_image.shape[0] % self.srf)] for hr_image in hr_images]
    hr_images = [hr_image if hr_image.shape[1] % self.srf == 0 else hr_image[:,:-(hr_image.shape[1] % self.srf)] for hr_image in hr_images]

    lr_images = [downscale(hr_image) for hr_image in hr_images]

    bicubic_images = [cv.resize(
      lr_image,
      (lr_image.shape[1] * self.srf, lr_image.shape[0] * self.srf), 
      interpolation=cv.INTER_CUBIC
    ) for lr_image in lr_images]

    # Choose random patches as basis for the dataset
    self.original_dataset = []
    for _ in range(N):
      image_index = np.random.randint(0, len(hr_images))
      hr_image, bicubic_image = hr_images[image_index], bicubic_images[image_index]
      x, y = np.random.randint(padding, hr_image.shape[1] - padding), np.random.randint(padding, hr_image.shape[0] - padding)
      bicubic_patch = bicubic_image[y-padding:y+padding+1, x-padding:x+padding+1].reshape(-1)
      center_pixel = hr_image[y, x] - bicubic_image[y, x]
      data = np.concatenate((bicubic_patch, [center_pixel]))
      self.original_dataset.append(data)

    self.original_dataset = np.unique(np.array(self.original_dataset), axis=0)
  
    if ACTIVE_SAMPLING_RATIO == 1:
      self.dataset = np.copy(self.original_dataset)
    else:
      distances = np.sum((self.original_dataset[:, None, :] - self.original_dataset[None, :, :])**2, axis=-1)
      np.fill_diagonal(distances, np.inf)
      self.bandwidth = SCALE_COEFFICIENT_BANDWIDTH * np.median(np.where(distances != np.inf))
      neighbor_indices = np.argsort(distances, axis=1)[:,:K]
      neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=-1)
      self.representativeness_scores = np.mean(np.exp(-neighbor_distances / (2*self.bandwidth)), axis=-1)

      self.available_dataset = np.copy(self.original_dataset)
      self.dataset = []
      for _ in range(floor(ACTIVE_SAMPLING_RATIO * N)):
        characteristic_scores = [self.get_characteristic_score(i, datapoint) for (i, datapoint) in enumerate(self.available_dataset)]
        max_characteristic_score_index = np.argmax(characteristic_scores)
        max_characteristic_score_datapoint = self.available_dataset[max_characteristic_score_index]
        self.available_dataset = np.delete(self.available_dataset, max_characteristic_score_index, axis=0)
        self.dataset.append(max_characteristic_score_datapoint)

    # Train the model
    self.train_x = torch.tensor(np.array([datapoint[:-1] for datapoint in self.dataset]))
    self.train_y = torch.tensor([datapoint[-1] for datapoint in self.dataset])

    torch.save(self.train_x, self.get_checkpoint_path("train_x"))
    torch.save(self.train_y, self.get_checkpoint_path("train_y"))

    if self.verbose: print("Training data set in {:.2f} seconds".format(time() - training_data_time))

  def upscale(self, image : np.ndarray) -> np.ndarray:
    if COLOR_SPACE == ColorSpace.YUV:
      y, u, v = cv.split(cv.cvtColor(image, cv.COLOR_BGR2YUV))
      y = self.upscale_channel(y).astype(np.uint8)
      u = cv.resize(u, (y.shape[1], y.shape[0]), interpolation=cv.INTER_CUBIC)
      v = cv.resize(v, (y.shape[1], y.shape[0]), interpolation=cv.INTER_CUBIC)
      return cv.cvtColor(cv.merge((y, u, v)), cv.COLOR_YUV2BGR)
    elif COLOR_SPACE == ColorSpace.GRAYSCALE:
      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
      gray = self.upscale_channel(gray)
      return gray

  def upscale_channel(self, image : np.ndarray) -> np.ndarray:
    image = image / 255.
    
    upscaled_image = cv.resize(image, (image.shape[1] * self.srf, image.shape[0] * self.srf), interpolation=cv.INTER_CUBIC)

    patches = []
    for i in range(padding, upscaled_image.shape[0] - padding):
      for j in range(padding, upscaled_image.shape[1] - padding):
        patches.append(upscaled_image[i-padding:i+padding+1, j-padding:j+padding+1].reshape(-1))

    test_x = torch.tensor(np.array(patches)).float()
    if torch.cuda.is_available():
      test_x = test_x.cuda()

    predictions = np.zeros((test_x.shape[0],))
    for i, test_x_batch in enumerate(torch.split(test_x, 1000)):
      predictions_batch = self.model(test_x_batch).mean.detach()

      if torch.cuda.is_available():
        predictions_batch = predictions_batch.cpu()
      
      predictions[i*1000:(i+1)*1000] = predictions_batch      


    predictions = predictions.reshape((upscaled_image.shape[0] - 2*padding, upscaled_image.shape[1] - 2*padding))
    upscaled_image[padding:-padding, padding:-padding] += predictions

    upscaled_image = self.iterative_back_projection(upscaled_image, image)
    upscaled_image = np.clip(upscaled_image, 0, 1)

    return upscaled_image * 255
  
  def iterative_back_projection(self, interpolated_image : np.ndarray, original_image : np.ndarray) -> np.ndarray:
    max_iterations = 50
    back_projection_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    current_image = interpolated_image.copy()
    current_error = np.inf

    for _ in range(max_iterations):
      downscaled = downscale(current_image)
      difference = original_image - downscaled
      if current_error - 0.01 <= np.linalg.norm(difference)**2: break
      current_error = np.linalg.norm(difference)**2

      upscaled_difference = cv.resize(difference, (current_image.shape[1], current_image.shape[0]), interpolation=cv.INTER_CUBIC)
      back_projection = cv.filter2D(upscaled_difference, -1, back_projection_kernel)
      current_image += back_projection * TAU

    return current_image

  def get_characteristic_score(self, i, datapoint) -> float:
    representativeness = self.representativeness_scores[i]
    diversity = self.get_diversity(datapoint)

    return CHARACTERISTIC_TRADEOFF * representativeness + (1 - CHARACTERISTIC_TRADEOFF) * diversity
  
  def get_diversity(self, datapoint) -> float:
    if len(self.dataset) == 0: return 0

    selected_distances = np.sum((self.dataset - datapoint)**2, axis=1)

    return np.min(-np.exp(-selected_distances / (2*self.bandwidth)))
  
  def get_training_images(self, image_type):
    if DATASET == TrainingDataset.Set14:
      image_paths = [f"Set14/image_SRF_{self.srf}/img_{image_number:03d}_SRF_{self.srf}_{image_type}.png" for image_number in range(1,15)]
    elif DATASET == TrainingDataset.BSD100:
      image_paths = load_dataset('eugenesiow/BSD100', split='validation')[image_type.lower()]

    images = [cv.imread(image_path) for image_path in image_paths]

    if COLOR_SPACE == ColorSpace.YUV:
      images = [cv.cvtColor(image, cv.COLOR_BGR2YUV)[:,:,0] / 255 for image in images]
    elif COLOR_SPACE == ColorSpace.GRAYSCALE:
      images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255 for image in images]
    return images
  
  def get_checkpoint_path(self, filename):
    return os.path.join(self.get_checkpoint_folder(), filename + ".pt")
  
  def get_checkpoint_folder(self):
    return f"checkpoints/{DATASET.name}/SRF_{self.srf}/"
  
def downscale(image : np.ndarray) -> np.ndarray:
  image = cv.GaussianBlur(image, (3,3), 1)
  return cv.resize(image, (image.shape[1] // SRF, image.shape[0] // SRF), interpolation=cv.INTER_CUBIC)

if __name__ == "__main__":
  sparse_gpr = AGPRSuperResolution(SRF, use_existing_model=False, verbose=True)

  for i in range(1,15):
    start_time = time()
    hr_image = cv.imread(f"Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_HR.png")
    lr_image = downscale(hr_image)
    interpolated_image = sparse_gpr.upscale(lr_image)
    cv.imwrite(f"Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_AGPR.png", interpolated_image)

  evaluation = Evaluator(SRF, image_nums=range(1,15), generate_gpr_images=False, verbose=True)
  evaluation.evaluate_method("AGPR")
  evaluation.evaluate_method("bicubic")
