# -*- coding: utf8 -*-
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

#pencil sketch  
def sketch(img): 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_blur = cv2.GaussianBlur(img_gray,(21,21),0,0) 
    img_blend = cv2.divide(img_gray,img_blur, scale = 236) 
    return img_blend 

#cartoon animation	
def mult_filtr(img): 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) 
    dst = cv2.filter2D(img,-1,kernel) 
    img_f = canny(dst,3750,3750,5) 
    #img_f = cv2.cvtColor(img_f,cv2.COLOR_BGR2RGB)
    return img_f 

#white contours
def canny(img,ch_min,ch_max, k): 
    img_canny = cv2.Canny(img,ch_min,ch_max, apertureSize = k) 
    img_mask = cv2.bitwise_not(img_canny) 
    img_res = cv2.bitwise_and(img,img, mask =img_mask) 
    return img_res

#image warping	
def warp(img,rows,cols):

    src_points = np.float32([[0,0], [cols-1,0], [0,rows -1], [cols-1,rows-1]])
    dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols), rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points,dst_points)
    img_output = cv2.warpPerspective(img,projective_matrix,(cols,rows))

    img_gray = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    ret,img_tresh = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY_INV)

    res_and = cv2.bitwise_and(img, img, mask = img_tresh )
    res_and = cv2.GaussianBlur(res_and,(71,71),0,0)
    f_img = cv2.add(img_output,res_and)
    #f_img = cv2.cvtColor(f_img,cv2.COLOR_BGR2RGB)
    return f_img
	
