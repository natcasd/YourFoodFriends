import numpy as np
import matplotlib.pyplot as plt
import cv2

npz = np.load('cropped_output.npz')

for img in npz['arr_0']:
    plt.imshow(img)
    plt.show()
# sample_image = npz['arr_0'][0]

# plt.imshow(sample_image)
# plt.show()
