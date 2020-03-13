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

#%% Question 4
import cv2
import numpy as np
One = cv2.imread("G:/Computer Vision/HW1_phase2/1.jpg")
Two = cv2.imread("G:/Computer Vision/HW1_phase2/2.jpg")
# changing to grayscale
One_Gray = cv2.cvtColor(One, cv2.COLOR_BGR2GRAY) 
Two_Gray = cv2.cvtColor(Two, cv2.COLOR_BGR2GRAY) 

#using sobel algorythm for 1.jpg
sobelx_One = cv2.Sobel(One_Gray,cv2.CV_64F,1,0,ksize=5) 
sobely_One = cv2.Sobel(One_Gray,cv2.CV_64F,0,1,ksize=5) 
sobel_One1 = np.sqrt(pow(sobelx_One, 2) + pow(sobely_One, 2))
sobel_One = sobel_One1.astype(np.uint16)

#using canny and LoG for 1.jpg
canny_One = cv2.Canny(One_Gray,100,200)
blur_One = cv2.GaussianBlur(One_Gray,(3,3),0)
laplacian_One = cv2.Laplacian(blur_One,cv2.CV_64F)

#using sobel algorythm for 2.jpg
sobelx_Two = cv2.Sobel(Two_Gray,cv2.CV_64F,1,0,ksize=5) 
sobely_Two = cv2.Sobel(Two_Gray,cv2.CV_64F,0,1,ksize=5) 
sobel_Two1 = np.sqrt(pow(sobelx_Two, 2) + pow(sobely_Two, 2))
sobel_Two = sobel_Two1.astype(np.uint16)

#using canny and LoG for 2.jpg
canny_Two = cv2.Canny(Two_Gray,100,200)
blur_Two = cv2.GaussianBlur(Two_Gray,(3,3),0)
laplacian_Two = cv2.Laplacian(blur_Two,cv2.CV_64F)
      
cv2.imshow('sobel_One',sobel_One) 
cv2.imshow('canny_One',canny_One) 
cv2.imshow("laplacian_One",laplacian_One) 

cv2.imshow('sobel_Two',sobel_Two) 
cv2.imshow('canny_Two',canny_Two) 
cv2.imshow("laplacian_Two",laplacian_Two) 

cv2.waitKey(0)
cv2.destroyAllWindows()


