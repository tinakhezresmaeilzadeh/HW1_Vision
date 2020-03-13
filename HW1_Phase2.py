# -*- coding: utf-8 -*-
"""
HW1_Phase2 Computer Vision
Dr. Mohammadzadeh

@author: Tina KhezrEsmaeilzadeh
@Student Number: 96101595

"""
#%% Question 3
import cv2
import numpy as np

Two = cv2.imread("G:/Computer Vision/HW1_phase2/2.jpg")
#resizing the image
cv2.resize(Two, (400, 300), interpolation=cv2.INTER_LINEAR)

#low pass filter
kernel_low_pass = np.ones((5,5), np.float32)/25
Low_Passed_Filter = cv2.filter2D(Two, -1, kernel_low_pass)
# changing to grayscale
Two_grayed = cv2.cvtColor(Two, cv2.COLOR_BGR2GRAY)

# kernels for edge detection
kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

#Sobel edge detection horizontal and vertical  
Sobelx = cv2.filter2D(Two_grayed, -1, kernel_horizontal)
Sobely = cv2.filter2D(Two_grayed, -1, kernel_vertical)

#total sobel edge detection
Sobel1 = np.sqrt(pow(Sobelx, 2) + pow(Sobely, 2))
Sobel = Sobel1.astype(np.uint16)

# high pass image
High_Passed_Filter = Two - Low_Passed_Filter  

cv2.imshow("low pass filter" , Low_Passed_Filter )
cv2.imshow("horizontal edge detection", Sobelx)
cv2.imshow("vertical edge detection", Sobely)
cv2.imshow("high pass filter" , High_Passed_Filter )
cv2.imshow("edge detection", Sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()


