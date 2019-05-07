"""
@author: arun.jose
@author: arpit.jain
"""

import os
import cv2



def apply_binary_mask_to_rgb(path1, path3):
    listing = os.listdir(path1)
    i = 0
    for file in listing:
        if file.startswith('.'):
            continue
        img = cv2.imread(path1 + '/' + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imwrite(path3 + '/_9_' + str(i) + ".png", res)
        i = i+1


path1 = "./color-asl-num-data"
path2 = './gray-asl-num-data'
path3 = './binary-asl-num-data'

apply_binary_mask_to_rgb(path1, path3)

