import os
import cv2

test_data_path = "Set14_smaller/"
if not os.path.exists(test_data_path):
  os.makedirs(test_data_path)

for i in range(1, 15):
  # Read the image
  filename = f"Set14/image_SRF_2/img_{i:03d}_SRF_2_HR.png"
  image = cv2.imread(filename)

  # Create data
  lr_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
  cv2.imwrite(test_data_path + f"{i:03d}_LR.png", lr_image)

  for srf in [2, 3, 4]:
    hr_image = cv2.resize(image, (lr_image.shape[1] * srf, lr_image.shape[0] * srf))
    cv2.imwrite(test_data_path + f"{i:03d}_HR_{srf}x.png", hr_image)

    bicubic_image = cv2.resize(lr_image, (lr_image.shape[1] * srf, lr_image.shape[0] * srf), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(test_data_path + f"{i:03d}_bicubic_{srf}x.png", bicubic_image)