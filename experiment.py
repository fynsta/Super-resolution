import cv2 as cv

from gprsr import GPRSR
from agpr import AGPRSuperResolution, downscale

agpr = AGPRSuperResolution(2, use_existing_model=True)
gprsr = GPRSR(2)

image1 = "Set14/image_SRF_3/img_001_SRF_3_HR.png"
hr_image1 = downscale(cv.imread(image1))
lr_image1 = downscale(hr_image1)
# srgpr_image1 = gprsr.apply(lr_image1)
agpr_image1 = agpr.upscale(lr_image1)

cv.imwrite("experiments/image1_hr.png", hr_image1)
cv.imwrite("experiments/image1_lr.png", lr_image1)
# cv.imwrite("experiments/image1_srgpr.png", srgpr_image1)
cv.imwrite("experiments/image1_agpr2.png", agpr_image1)

image2 = "Set14/image_SRF_3/img_002_SRF_3_HR.png"
hr_image2 = downscale(cv.imread(image2))
lr_image2 = downscale(hr_image2)
# srgpr_image2 = gprsr.apply(lr_image2)
agpr_image2 = agpr.upscale(lr_image2)

cv.imwrite("experiments/image2_hr.png", hr_image2)
cv.imwrite("experiments/image2_lr.png", lr_image2)
# cv.imwrite("experiments/image2_srgpr.png", srgpr_image2)
cv.imwrite("experiments/image2_agpr2.png", agpr_image2)




