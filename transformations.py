import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from glob import glob
from PIL import Image

folder_dir = "C:/Users/Sean/RAINN/examples/untouched"

for images in os.listdir(folder_dir):
    if images.endswith(".jpg"):
        for i in range(len(images)):
            image = cv2.imread(images)
        #kernel = np.ones((5,5), np.uint8)
        kernel = np.ones((1,1), np.uint8)

        plt.figure(figsize=(20, 20))
        plt.subplot(3, 2, 1)
        plt.title("Original")
        plt.imshow(image)

        #erosion = cv2.erode(image, kernel, iterations = 1)
        #dilation = cv2.dilate(image, kernel, iterations = 1)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        h, w = image.shape[:2]
        cX, cY = (w // 2, h // 2)
        R = cv2.getRotationMatrix2D((cX, cY), 1, 1.0)
        rotated = cv2.warpAffine(image, R, (w, h))
        
        #noise_reduced = cv2.fastNlMeansDenoisingColored(rotated, None, 10, 10, 7, 15)
        eroded = cv2.erode(rotated, kernel, iterations = 1)
        final = eroded
        img = cv2.imwrite('C:/Users/Sean/RAINN/examples/transformations/' + str(images).replace(".jpg", 'rs') + '.jpg', final)




#for trans in os.listdir(folder_dir):
#    if trans.endswith(".jpg"):
        

        

        # dilation = cv2.dilate(image, kernel, iterations = 1)
        # plt.subplot(3, 2, 3)
        # plt.title("Dilation")
        # plt.imshow(dilation)

        # cv2.imwrite("C:/Users/Sean/RAINN/examples/dilation.jpg", dilation)

        # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # plt.subplot(3, 2, 4)
        # plt.title("Opening")
        # plt.imshow(opening)

        # cv2.imwrite("C:/Users/Sean/RAINN/examples/opening.jpg", opening)

        # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # plt.subplot(3, 2, 5)
        # plt.title("Closing")
        # plt.imshow(closing)

        # cv2.imwrite("C:/Users/Sean/RAINN/examples/closing.jpg", closing)