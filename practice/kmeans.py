import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("neutrophils.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 20 array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]


# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
plt.figure('Kmeans Image')
plt.subplot(211)
plt.title('original image')
plt.imshow(image)
plt.subplot(212)
plt.title('segmented_image')
plt.imshow(segmented_image)
plt.show