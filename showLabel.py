import cv2
import os
import numpy as np

abjads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Initialize a list to store images and labels
images = []
labels = []

for folder_dataset in abjads:
    for filename in os.listdir(f'data-test/{folder_dataset}'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read image
            image = cv2.imread(os.path.join(f'data-test/{folder_dataset}', filename))
            images.append(image)
            labels.append(folder_dataset)
            break  # Only take one image from each folder

# Resize images to the same size
resized_images = [cv2.resize(img, (100, 100)) for img in images]

# Add labels to images
labeled_images = []
for img, label in zip(resized_images, labels):
    labeled_img = img.copy()
    cv2.putText(labeled_img, label, (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    labeled_images.append(labeled_img)

# Create a grid of 4x4 images
grid_size = 8
grid_images = []

for i in range(0, len(labeled_images), grid_size):
    row_images = labeled_images[i:i + grid_size]
    if len(row_images) < grid_size:
        # Add black images to fill the row if there are not enough images
        row_images += [np.zeros((100, 100, 3), dtype=np.uint8)] * (grid_size - len(row_images))
    grid_images.append(np.hstack(row_images))

# Combine rows into a single image
combined_image = np.vstack(grid_images)

# Display the combined image
cv2.imshow('Combined Image with Labels', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()