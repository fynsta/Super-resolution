from collections import defaultdict
import cv2
import os
from skimage import metrics
from gp_models import RBFModel, ExponentialModel, Matern32Model, Matern52Model, SpectralMixtureModel

def get_metrics(hrImgGray, upsampledImgGray, name):
  psnr = metrics.peak_signal_noise_ratio(hrImgGray, upsampledImgGray)
  ssim = metrics.structural_similarity(hrImgGray, upsampledImgGray)
  print(f'{name} - PSNR: {psnr:.3f}, SSIM: {ssim:.3f}')
  return (psnr, ssim)

def evaluate_image(i, srf):
  values = dict()

  print(f'Image {i}')
  hrImagePath = f'Set14/image_SRF_{srf}/img_{i:03d}_SRF_{srf}_HR.png'
  hrImg = cv2.imread(hrImagePath)
  hrImgGray = cv2.cvtColor(hrImg, cv2.COLOR_BGR2GRAY)[srf:-srf, srf:-srf]

  bicubicImagePath = f'Set14/image_SRF_{srf}/img_{i:03d}_SRF_{srf}_bicubic.png'
  bicubicImg = cv2.imread(bicubicImagePath)
  bicubicImgGray = cv2.cvtColor(bicubicImg, cv2.COLOR_BGR2GRAY)[srf:-srf, srf:-srf]
  metrics = get_metrics(hrImgGray, bicubicImgGray, 'Bicubic (Baseline)')
  values['Bicubic (Baseline)'] = metrics

  for model in [RBFModel, ExponentialModel, Matern32Model, Matern52Model, SpectralMixtureModel]:
    gprImagePath = f'Set14/image_SRF_{srf}/img_{i:03d}_SRF_{srf}_GPR_{model._get_name()}.png'
    if not(os.path.exists(gprImagePath)):
      continue

    gprImg = cv2.imread(gprImagePath)
    gprImgGray = cv2.cvtColor(gprImg, cv2.COLOR_BGR2GRAY)[srf:-srf, srf:-srf]
    
    metrics = get_metrics(hrImgGray, gprImgGray, model._get_name())
    values[model._get_name()] = metrics
  print()
  return values

def evaluate(srf):
  metrics = defaultdict(lambda: (0,0))

  for i in range(1, 15):
    image_metrics = evaluate_image(i, srf)
    for key, value in image_metrics.items():
      previous_ssims, previous_psnrs = metrics[key]
      metrics[key] = (previous_ssims + value[1], previous_psnrs + value[0])
  
  print('Averages')
  for key, value in metrics.items():
    print(f'{key} - PSNR: {value[0]/14:.3f}, SSIM: {value[1]/14:.3f}')

