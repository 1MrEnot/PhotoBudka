# -*- coding: utf8 -*-

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

import filters as filt
import moustGlass as mG


cap = cv2.VideoCapture(1)

moustache = cv2.imread('Images/Moustache.jpg', 0)
glass = cv2.imread('Images/Glass.png',0)

font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier('Haare/haarcascade_frontalface_default.xml')

old_faces = []
timer = 7
x = 0

def f_check(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_rects = face_cascade.detectMultiScale(frame_gray, 1.5, 6)
    if ( face_rects != ()):
        faces = 1
        return faces
    else:
        return 0
    
n_count_const = 100 
i_count = 0 

read_face = True 
effects = 0 

#main cycle
if __name__ == "__main__":

	while True:
		ret, frame = cap.read()
		frame = cv2.flip(frame,1)
		rows,cols = frame.shape[:2]
		if read_face: 
			effects = f_check(frame)

			if effects > 0:
				read_face = False
			else:
				cv2.putText(frame, 'Come to camera plz' ,(10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,(128,200,255),2)
		#using filters 
		else:
			#sketch 
			if (effects == 1):
				frame  = filt.sketch(frame)
			#glass  
			elif (effects == 2):
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
				faces = face_cascade.detectMultiScale(frame_gray, 1.5, 4)
				faces = mG.SmallFacesCheck(faces, frame_gray, 0.3)
				faces = mG.FiFCheck(faces)
				res = mG.glassing(frame,faces,glass)
				frame = res
			#cartoon filter
			elif (effects == 3):
				frame  = filt.mult_filtr(frame)
			#moustache
			elif (effects == 4):
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
				faces = face_cascade.detectMultiScale(frame_gray, 1.5, 4)
				faces = mG.SmallFacesCheck(faces, frame_gray, 0.3)
				faces = mG.FiFCheck(faces)
				res = mG.moustacheing (frame, faces, moustache)
				frame = res

			#warping
			elif (effects == 5):
				frame  = filt.warp(frame,rows,cols)

			#filter counter
			i_count += 1 
			if i_count == 100:
				effects +=1
				i_count = 0
			if effects > 5: 
				i_count = 0 
				read_face = True
				effects = 0
				
		cv2.imshow('frame', frame) 
		c = cv2.waitKey(1)
	 
		if c == 27:
			break


	cap.release()
	cv2.destroyAllWindows()