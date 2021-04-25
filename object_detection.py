# %%
# import the necessary packages
import argparse
import random
import time
import cv2
from skimage.io import imread, imshow, imsave
import numpy as np
import os
from skimage.transform import rescale, resize, downscale_local_mean
base = "Images/"
from tensorflow import keras
model = keras.models.load_model('cnn_model_2')


def segment_red(image_bgr):

	img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	lower_red = np.array([0,45,45])
	upper_red = np.array([11,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	lower_red = np.array([170,45,45])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
	# join two masks
	mask = mask0 + mask1
	# set output to zero everywhere except mask
	output_img = image_bgr.copy()
	output_img[np.where(mask==0)] = 0
	# or your HSV image, which I *believe* is what you want
	output_hsv = image_bgr.copy()
	output_hsv[np.where(mask==0)] = 0
	output_bgr=cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
	#output_rgb_=cv2.cvtColor(output_hsv, cv2.COLOR_BGR2RGB)

	return output_bgr

def get_proposed_rectangles(binary_img):
	contours = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	rects = []
	for cntr in contours:
		rects.append(cv2.boundingRect(cntr))
	return rects

def binarize_segments(image_bgr):
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 3)
	thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
	return thresh

def detect_road_signs(model, image_bgr, rects, confidence=0.9, max_obj_per_image=2, save=False, path=None):
	output_w_all_rects = image_bgr.copy()
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	final_output = image_bgr.copy()
	probs, recs= [], []
	rect_j = 0
	for (x, y, w, h) in rects:
		ratio =float(max(w, h))/float(min(w, h))
		if w*h > 300 and ratio < 1.75:
			crop_img_ = resize(image_rgb[y:y+h, x:x+w], (100, 100))/255.0
			crop_img_ = np.array([crop_img_])
			label = model.predict(crop_img_)
			probs.append(label)
			recs.append((x, y, w, h))
		color = (255, 0, 0)
		cv2.rectangle(output_w_all_rects, (x, y), (x + w, y + h), color, 4)
	probs, recs = zip(*sorted(zip(probs, recs), reverse=True))
	detected_signs = 0
	for i in range(min(max_obj_per_image, len(probs))):
		x, y, w, h = recs[i]
		if probs[i] > confidence:
			color = (0, 255, 0)
			final_output = cv2.rectangle(final_output, (x, y), (x + w, y + h), color, 4)
			detected_signs +=1
	if save is True:
		if path is not None:
			imsave(path, final_output)

	return output_w_all_rects, final_output, detected_signs, rect_j
