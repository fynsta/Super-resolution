import os
import cv2 as cv
import numpy as np
from evaluation import Evaluator, PerceptualSimilarityMetric
from main import SRF, USED_COLOR_SPACE, IMAGE_NUMS, GPRSR
from gp_models import GeneralModel, RBFModel, ExponentialModel, Matern32Model, Matern52Model, SpectralMixtureModel, PeriodicModel, 


class AutomaticModelConstructor():
  def __init__(self, base_models, verbose=False):
    self.verbose = verbose
    self.evaluator = Evaluator(SRF)

    self.base_models = base_models

    self.find_best_start_model()
    
  def find_best_start_model(self):
    self.current_best_ssim = 0
    self.current_best_model = None

    
    for base_model in self.base_models:
      gprsr = GPRSR(SRF, base_model, USED_COLOR_SPACE)

      for i in IMAGE_NUMS:
        lrImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_LR.png'
        gprImagePath = f'Set14/image_SRF_{SRF}/img_{i:03d}_SRF_{SRF}_GPR_{base_model._get_name()}.png'
        if os.path.exists(gprImagePath):
          continue

        lrImg = cv.imread(lrImagePath)
        gprImg = gprsr.apply(lrImg)

        cv.imwrite(gprImagePath, gprImg)

      ssim = self.evaluator.evaluate_metric(base_model, PerceptualSimilarityMetric.SSIM)
      if self.verbose:
        print(f'{base_model._get_name()} - SSIM: {ssim:.3f}')

      if ssim > self.current_best_ssim:
        self.current_best_ssim = ssim
        self.current_best_model = base_model

    if self.verbose:
      print(f'Best model: {self.current_best_model._get_name()}')

  def step(self):

  

if __name__ == '__main__':
  AutomaticModelConstructor([RBFModel, ExponentialModel, Matern32Model, Matern52Model, SpectralMixtureModel, PeriodicModel], verbose=True)