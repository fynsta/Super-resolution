from enum import Enum
import numpy as np

class PatchInterpolationMode(Enum):
  AVERAGE = 0
  BILINEAR = 1

# Generates patches from the given images, and provides an iterator over them.
# You can pass in a list of images, and it will generate patches from all of them.
class PatchHandler:
  def __init__(
      self, 
      images, 
      base_patch_size = 20,
      stride_percentage = 0.9
    ):
    self.image_shape = images[0].shape
    assert base_patch_size <= self.image_shape[0] and base_patch_size <= self.image_shape[1], f'Patch size {base_patch_size} is too big for image size {self.image_shape}'
    assert np.all([aspect_ratio(image.shape) == aspect_ratio(self.image_shape) for image in images]), 'All images must have the same aspect ratio'
    assert np.all([image.shape[0] >= self.image_shape[0] for image in images]), 'All further images must be at least as big as the first image'

    self.images = images
    self.scale_factors = [int(image.shape[0] / self.image_shape[0]) for image in images]
    self.base_patch_size = base_patch_size

    # The stride is the distance between two patches, both horizontally and vertically.
    # Note that we need to subtract 2 from the patch size to account for the edges (which are not predicted).
    # The paper uses an overlap of 66% overall (which means a stride rate of around 0.8 according to my rough calculations)
    self.stride = int((base_patch_size-2) * stride_percentage)

    self._generate_patch_locations()

  def _generate_patch_locations(self):
    # Calculating the indices of the patches to be used for training.
    # Patches that would go out of bounds are not used are moved to the interior so that they are within bounds.
    self.patch_locations = []
    for x in range(0, self.image_shape[0], self.stride):
      x = min(x, self.image_shape[0] - self.base_patch_size)
      for y in range(0, self.image_shape[1], self.stride):
        y = min(y, self.image_shape[1] - self.base_patch_size)
        self.patch_locations.append((x, y))

  # Rebuilds the image from the patches, using the given patch interpolation method for overlapping patches.
  def combine_patches(self, patches, patch_interpolation = PatchInterpolationMode.AVERAGE):
    scale_factor = int(patches[0].shape[0] / self.base_patch_size)

    upscaled_patch_locations = [(x * scale_factor, y * scale_factor) for x, y in self.patch_locations]
    upscaled_patch_size = self.base_patch_size * scale_factor
    upscaled_shape = (self.image_shape[0] * scale_factor, self.image_shape[1] * scale_factor)

    result = np.zeros(upscaled_shape)
    weights = np.zeros(upscaled_shape)

    for (x, y), patch in zip(upscaled_patch_locations, patches):
      # Considerations for edges. Since they are not predicted by the model, they should not be used.
      # Except at the edge of the image, where they are used to fill in the missing pixels.
      # These values are straight from the original (bicubic) upscaling.
      x_start = 0 if x == 0 else 1
      y_start = 0 if y == 0 else 1
      x_end = upscaled_patch_size if x + upscaled_patch_size == upscaled_shape[0] else upscaled_patch_size - 1
      y_end = upscaled_patch_size if y + upscaled_patch_size == upscaled_shape[1] else upscaled_patch_size - 1

      for i in range(x_start, x_end):
        for j in range(y_start, y_end):
          if patch_interpolation == PatchInterpolationMode.AVERAGE:
            weight = 1
          elif patch_interpolation == PatchInterpolationMode.BILINEAR:
            vertical_weight = 1 - abs(1/2 - i / upscaled_patch_size)
            horizontal_weight = 1 - abs(1/2 - j / upscaled_patch_size)
            weight = vertical_weight * horizontal_weight
          
          result[x + i, y + j] += patch[i, j] * weight
          weights[x + i, y + j] += weight
      
    result /= weights
    return result

  def __iter__(self, image_indices = None):
    image_indices = image_indices if image_indices is not None else range(len(self.images))
    images = [self.images[i] for i in image_indices]
    scale_factors = [self.scale_factors[i] for i in image_indices]
    return PatchIterator(images, scale_factors, self.base_patch_size, self.patch_locations)
  
class PatchIterator:
  def __init__(self, images, scale_factors, base_patch_size, patch_locations):
    self.images = images
    self.scale_factors = scale_factors
    self.base_patch_size = base_patch_size
    self.patch_locations = patch_locations
    self.current_patch = 0
  
  def __next__(self):
    if self.current_patch >= len(self.patch_locations):
      raise StopIteration
    
    base_location = self.patch_locations[self.current_patch]
    patches = []
    
    for image, scale_factor in zip(self.images, self.scale_factors):
      patch_size = int(self.base_patch_size * scale_factor)
      location = (int(base_location[0] * scale_factor), int(base_location[1] * scale_factor))
      patch = image[location[0]:location[0]+patch_size, location[1]:location[1]+patch_size].copy()

      patches.append(patch)

    self.current_patch += 1
    return patches

def aspect_ratio(tuple):
  return tuple[0] / tuple[1]