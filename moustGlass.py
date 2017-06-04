# -*- coding: utf8 -*-

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

#moustache addtion
def moustacheing (img, faces, moustache):
    
    if len(faces) == 0:
        return(img)
    
    res = img[:]

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for (x, y, w, h) in faces:
        new_moustache = []
        cut = res[x:x+w, y:y+h]
        cut_gray = img_gray[x:x+w, y:y+h]

        moustache = cv2.resize(moustache, (w, h), interpolation = cv2.INTER_AREA)
        
        num_cols, num_rows = moustache.shape[:2]
        src_points = np.float32([[0,0], [moustache.shape[0],0], [0,moustache.shape[1]], [moustache.shape[0],moustache.shape[0]]])
        dst_points = np.float32([[int(moustache.shape[1]*0.14),int(moustache.shape[0]*0.65)], [int(moustache.shape[1]*0.86),int(moustache.shape[0]*0.65)], [int(moustache.shape[1]*0.14),int(moustache.shape[0]*0.80)], [int(moustache.shape[1]*0.86),int(moustache.shape[0]*0.80)]])
        translation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        new_moustache = cv2.bitwise_not(cv2.warpPerspective(cv2.bitwise_not(moustache), translation_matrix, (num_rows, num_cols)))
        res[y:y+w, x:x+h] = cv2.bitwise_and(res[y:y+w, x:x+h], cv2.cvtColor(new_moustache, cv2.COLOR_GRAY2RGB), mask = new_moustache)
        
    return res

#glass addtion
def glassing (img, faces, glass):
    
    if len(faces) == 0:
        return(img)
    
    res = img[:]

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for (x, y, w, h) in faces:
        new_glass = []
        cut = res[x:x+w, y:y+h]
        cut_gray = img_gray[x:x+w, y:y+h]

        glass = cv2.resize(glass, (w, h), interpolation = cv2.INTER_AREA)
        
        num_cols, num_rows = glass.shape[:2]
        src_points = np.float32([[0,0], [glass.shape[0],0], [0,glass.shape[1]], [glass.shape[0],glass.shape[0]]])
        dst_points = np.float32([[int(glass.shape[1]*0.10),int(glass.shape[0]*0.25)], [int(glass.shape[1]*0.90),int(glass.shape[0]*0.25)], [int(glass.shape[1]*0.10),int(glass.shape[0]*0.48)], [int(glass.shape[1]*0.90),int(glass.shape[0]*0.48)]])
        translation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        new_glass = cv2.bitwise_not(cv2.warpPerspective(cv2.bitwise_not(glass), translation_matrix, (num_rows, num_cols)))
        _, mask = cv2.threshold(new_glass, 127,255,cv2.THRESH_BINARY)
        res[y:y+w, x:x+h] = cv2.bitwise_and(res[y:y+w, x:x+h], cv2.cvtColor(new_glass, cv2.COLOR_GRAY2RGB), mask = mask)
        
    return res

#removing small faces
def SmallFacesCheck (faces_list, img, hp = 0.2):

    min_sq = int((img.shape[0]*hp)**2)
    faces_good_list = []
    for i in range (len(faces_list)):
        if faces_list[i,2]*faces_list[i,3] > min_sq:
            faces_good_list.append(list(faces_list[i]))
    return faces_good_list

#removing double-detected faces
def FiFCheck2 (sm_list):
    
    if len(sm_list) < 2:
        return sm_list
    
    good_list = []
    n = len(sm_list)
    for i in range (n):
        iList = sm_list[i]
        for j in range (i+1, n):
            jList = sm_list[j]
            if ((iList[0]<jList[0]+jList[3] or iList[0]>jList[0]) or (iList[1]<jList[1]+jList[3] or iList[1]>jList[1])):
                if jList[3]>iList[3]:
                    good_list.append(iList)
                else:
                    good_list.append(jList)
    return good_list
	
