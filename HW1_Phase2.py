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

#%% Question 6
import cv2
import numpy as np

Three = cv2.imread("G:/Computer Vision/HW1_phase2/3.jpg",cv2.IMREAD_GRAYSCALE)
Parameters = cv2.SimpleBlobDetector_Params()

Parameters.minThreshold = 20
Parameters.thresholdStep = 10
Parameters.maxThreshold = 300

Parameters.filterByCircularity = True
Parameters.minCircularity = 0.1

Parameters.filterByArea = True
Parameters.minArea = 1000

Parameters.filterByConvexity = True
Parameters.minConvexity = 0.7

Parameters.filterByInertia = True
Parameters.minInertiaRatio = 0.05

d = cv2.SimpleBlobDetector_create(Parameters).detect(Three)
Final_image = cv2.drawKeypoints(Three, d, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Three", Final_image )
cv2.waitKey(0)
cv2.destroyWindow("Three")



#%% 6 prime
import cv2
import numpy as np

img = cv2.imread("G:/Computer Vision/HW1_phase2/3.jpg", 0)
# using a lowpass filter
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#finding circles
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=40,param2=13,minRadius=0,maxRadius=30)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # drawing the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,0,255),1)
    # drawing the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Question 3 phase 2 without Gaussian Filter
import cv2
import numpy as np
# ready for capturing
cap = cv2.VideoCapture("G:/Computer Vision/HW1_phase2/MyVideo.avi")
while(1):
    #reading frames
  
    _, frame1 = cap.read() 
    
      
    #changing frames to grayscale
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) 
    # using Sobel    
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5) 
       
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5) 
    
    sobel1 = np.sqrt(pow(sobelx, 2) + pow(sobely, 2))
    sobel = sobel1.astype(np.uint16)
    # using prewitt filter
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    
    prewittx = cv2.filter2D(frame, -1, kernelx)
    prewitty = cv2.filter2D(frame, -1, kernely)
    
    prewitt = prewittx + prewitty
    #using canny filter    
    canny = cv2.Canny(frame,100,200)
      
     
    cv2.imshow('prewitt',prewitt) 
    cv2.imshow('canny',canny) 
    cv2.imshow('sobel',sobel)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
  
cv2.destroyAllWindows() 
cap.release() 


#%% Question 3 phase 2 with Gaussian Filter

import cv2
import numpy as np
# ready for capturing
cap = cv2.VideoCapture("G:/Computer Vision/HW1_phase2/MyVideo.avi")
while(1): 
    #reading frames
  
    _, frame1 = cap.read() 
      
    #changing frames to grayscale
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) 
    #using gaussian filter
    frame_gaussian = cv2.GaussianBlur(frame,(5,5),0)
    # using Sobel   
    sobelx = cv2.Sobel(frame_gaussian,cv2.CV_64F,1,0,ksize=5)       
    sobely = cv2.Sobel(frame_gaussian,cv2.CV_64F,0,1,ksize=5) 
    
    sobel1 = np.sqrt(pow(sobelx, 2) + pow(sobely, 2))
    sobel = sobel1.astype(np.uint16)

    
    
    # using prewitt filter
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    
    prewittx = cv2.filter2D(frame_gaussian, -1, kernelx)
    prewitty = cv2.filter2D(frame_gaussian, -1, kernely)
    
    prewitt1 = np.sqrt(pow(prewittx, 2) + pow(prewitty, 2))
    prewitt = prewitt1.astype(np.uint8)
    
    
    #using canny filter  
    canny = cv2.Canny(frame_gaussian,100,200)
      
    
    cv2.imshow('prewitt',prewitt) 
    cv2.imshow('canny',canny) 
    cv2.imshow('sobel',sobel) 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
  
cv2.destroyAllWindows() 
cap.release() 

