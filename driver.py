"""
@author: arun.jose
@author: arpit.jain
"""

import cv2
import numpy as np
import time
import roi_processor
import convolutional_nn as cnn


font = cv2.FONT_HERSHEY_SIMPLEX

debugMode = False


def asl_to_text():
    global debugMode, font

    # Load the CNN model
    trained_model = cnn.load_conv_neural_net()

    # Get the camera input
    captured_frame = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    if debugMode:
        cv2.namedWindow('Dilated Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Eroded Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Filtered Image', cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = ""
    start_time = time.time()

    while True:

        # Capture frames from the camera
        read_success, current_frame = captured_frame.read()

        # Flip the image (across y-axis. +ve value: 1 just for y-axis flip.)
        current_frame = cv2.flip(current_frame, 1)

        # Blur the image
        blur = cv2.blur(current_frame, (3, 3))

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a mask/binary image of skin region
        # White -> corresponds to skin colors
        # Black -> Rest of the region
        skin_region_mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

        # Kernel matrices for morphological transformation
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Perform morphological transformations to filter out the background noise
        # Dilation increase skin color area
        # Erosion removes extra skin color area created by dilation or noise
        # Median filter used to remove salt and pepper noise due to background.
        dilation = cv2.dilate(skin_region_mask, kernel_ellipse, iterations=1)
        erosion = cv2.erode(dilation, kernel_square, iterations=1)
        filtered = cv2.medianBlur(erosion, 5)
        median = filtered

        if debugMode:
            cv2.imshow('Dilated Image', dilation)
            cv2.imshow('Eroded Image', erosion)
            cv2.imshow('Filtered Image', filtered)

        # Conversion from Grayscale to Binary
        # Threshold to set pixel value
        ret, thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)

        # Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with maximum area
        max_area = 100
        max_contour_index = 0
        for i in range(len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour_index = i

        # If no contours detected, continue to next frame
        if len(contours) is 0:
            continue

        # Largest area contour
        largest_contour = contours[max_contour_index]

        # Print bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        hand_image = current_frame.copy()[y:y + h + 50, x:x + w + 50]
        # Resizing to img_rows x img_cols size, which is the same as training dataset
        # TODO: Future work to come up with better approach for this, as resizing causes
        # TODO: several loss of features.
        hand_resized_image = cv2.resize(hand_image, (cnn.img_rows, cnn.img_cols))

        if read_success:
            roi = roi_processor.binarize_and_classify_roi(hand_resized_image, frame_count, trained_model)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('ROI', roi)

            frame_count = frame_count + 1
            end_tim = time.time()
            time_diff = (end_tim - start_time)
            if time_diff >= 2:
                fps = 'FPS:%s' % frame_count
                start_time = time.time()
                frame_count = 0

        # cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
        cv2.putText(current_frame, fps, (10, 20), font, 0.7, (0, 255, 0), 2, 1)

        cv2.imshow('Original', current_frame)

        plot = np.zeros((512, 512, 3), np.uint8)
        plot = cnn.display_histogram(plot)
        cv2.imshow('ASL to Text', plot)

        # Exit by pressing 'ESC'
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break

    captured_frame.release()
    cv2.destroyAllWindows()


# Call ASL to Text in the driver program
asl_to_text()


