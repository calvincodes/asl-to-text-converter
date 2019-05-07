"""
@author: arun.jose
@author: arpit.jain
"""

import cv2
import threading
import convolutional_nn as cnn


def binarize_and_classify_roi(roi, frame_count, model):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, result = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Provide a window of 5 frames for user to change the gesture and system to predict the previous one
    if (frame_count % 5) == 4:
        t = threading.Thread(target=cnn.classify_asl_symbol, args=[model, result])
        t.start()

    return result


