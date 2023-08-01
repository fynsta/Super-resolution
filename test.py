import numpy as np
image = np.zeros((20, 20), dtype=np.uint8)

# Use bilinear filter where center is most important
for i in range(20):
  for j in range(20):
    image[i, j] = 