import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import cv2
import shutil
import os
import time

st = time.time()
image_list = []
only_filenames = []
for dir_path, _, filenames in os.walk('./cluster_revised_Sep_2'):
    for filename in filenames:
        if filename.endswith('.png'):
            image_list.append(dir_path.replace('\\', '/') + '/' + filename)
            only_filenames.append(filename)

print('image list,', len(image_list))
# print(image_list)


# Create directory
if os.path.exists('./revised_clusters'):
    shutil.rmtree('./revised_clusters')

os.makedirs('./revised_clusters', exist_ok=True)
os.makedirs('./revised_clusters/multi', exist_ok=True)

def compare_images(image1, image2):
    image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    hist_image1 = cv2.calcHist([image1_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_image2 = cv2.calcHist([image2_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist_image1, hist_image1)
    cv2.normalize(hist_image2, hist_image2)
    similarity = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CORREL)

    return similarity


def extract_channel1_images(image):
    # Remove dark edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold for dark regions
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # Non zero dark pixels
    coords = cv2.findNonZero(thresh)
    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)
    # Crop
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def save_cropped_image(image, path):
    cv2.imwrite(filename=path, img=image)


def plot_image_and_cropped(image, cropped):
    plt.figure(figsize=(10, 5))  # Set the figure size
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Original Image: {image.shape}')
    plt.axis('off')
    # Cropped image
    plt.subplot(1, 2, 2)
    plt.imshow(cropped)
    plt.title('Cropped Image')
    plt.axis('off')
    plt.show()


for i, img in enumerate(image_list):
    # Removing particles
    if 'particles' not in img:
        im = cv2.imread(img)
        horizontal_split = im.shape[1] // 6
        channel1_region = im[:, 0:horizontal_split]
        channel2_region = im[:, horizontal_split:horizontal_split * 2]
        channel3_region = im[:, horizontal_split * 2:horizontal_split * 3]
        channel7_region = im[:, horizontal_split * 3:horizontal_split * 4]
        # Skipping the grayscale channel in the middle
        channel11_region = im[:, horizontal_split * 5:horizontal_split * 6]

        cropped_channel1 = extract_channel1_images(channel1_region)
        # cropped_channel2 = extract_channel1_images(channel2_region)

        similarity_score = compare_images(channel1_region, channel11_region)
        if similarity_score < 0.2:
            save_cropped_image(cropped_channel1, f'./revised_clusters/multi/chan1_{only_filenames[i]}')
            save_cropped_image(channel2_region, f'./revised_clusters/multi/chan2_{only_filenames[i]}')
            save_cropped_image(channel3_region, f'./revised_clusters/multi/chan3_{only_filenames[i]}')
            save_cropped_image(channel7_region, f'./revised_clusters/multi/chan7_{only_filenames[i]}')
            save_cropped_image(channel11_region, f'./revised_clusters/multi/chan11_{only_filenames[i]}')

        # plot_image_and_cropped(channel1_region, cropped_channel1)
        # plot_image_and_cropped(channel11_region, im)

print('Time taken:', time.time() - st)
