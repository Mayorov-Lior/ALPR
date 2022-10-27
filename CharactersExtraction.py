import sys

import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import try_all_threshold, threshold_isodata, threshold_li, threshold_mean, threshold_minimum
from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola
import numpy as np
import tensorflow as tf
import random
import cv2
from PlateExtractionSeg import createUnetModel
import re
import math

from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils
import os

minCharW = 40
minPlateW = 60
minPlateH = 20
numChars = 8
#sys.path.append("./drive/My Drive/lp_data")

show_characters = False
show_contours = False
#PATH = '/content/drive/My Drive/lp_data/'
PATH = 'C:/Users/Lior/Documents/work/License Plate Recognition/'
hNew = 250
def adjustToUInt8(img):
  if (np.amax(img) <= 1):
    img = (img*255)
  img = img.astype(np.uint8)
  return img

def sharpenImg(img, amount):
    kernel = amount * np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)  
    return img

def changeContrast(img):
    contrastFactor = 2.8
    contrast = tf.image.adjust_contrast(img, contrast_factor=contrastFactor)
    sess = tf.InteractiveSession()
    contrast = contrast.eval()
    sess.close() 
    return contrast

'''def getThImg(img, thFunc):
  thNumber = thFunc(img)
  _, thImg = cv2.threshold(img,thNumber,255,cv2.THRESH_BINARY)
  return thImg'''

def getThImg(img, th):
  _, thImg = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
  return thImg


def proccessLP(lp): # prepare lp to for char detection
  lp = resizeLP(cv2.cvtColor(lp.astype(np.uint8), cv2.COLOR_BGR2GRAY))
  lp = lp.astype(np.uint8)
  equ = cv2.equalizeHist(lp)
  equ = equ.astype(np.uint8)

  thresholds = [30, 60, 80, 100, 110, 117, 126, 133, 140, 150, 165, 185, 200, 230]
  lps = [getThImg(lp, th) for th in thresholds] + [getThImg(equ, th) for th in thresholds] 

  return lps

'''def proccessLP(lp): # prepare lp for char detection
  
  lp = resizeLP(cv2.cvtColor(lp.astype(np.uint8), cv2.COLOR_BGR2GRAY))
  lp = lp.astype(np.uint8)
  equ = cv2.equalizeHist(lp)
  equ = equ.astype(np.uint8)

  thresholds = [lambda x: 30, lambda x: 60,lambda x: 80,lambda x: 100,lambda x: 120,lambda x: 130,lambda x: 140,lambda x: 150,
  lambda x: 180,lambda x: 200,lambda x: 210,lambda x: 240]

  kernel = lambda s : np.ones((s,s),np.uint8)
  #lp = cv2.erode(lp,kernel)
  #equ = cv2.erode(equ,kernel)
  lps = [getThImg(lp, thFunc) for thFunc in thresholds] + [getThImg(equ, thFunc) for thFunc in thresholds] 

  #lps = [cv2.medianBlur(l, 3) for l in lps]
  #lps = [cv2.dilate(l,kernel(3),iterations = 1) for l in lps]
  #lps = [cv2.erode(l,kernel(3),iterations = 1) for l in lps]
  #lps = [l for l in lps if (cv2.countNonZero(l) < 0.7*imageArea(l.shape)) and (cv2.countNonZero(l) > 0.3*imageArea(l.shape)) ] 
  
  for l in lps:
    cv2.imshow('processLP lp!', l)
    cv2.waitKey(0)

  return lps'''

def resizeLP(lp):
  h,w = lp.shape[:2]
  wNew = int((w/(h*1.0))*hNew)
  lp = cv2.resize(lp, (wNew, hNew),cv2.INTER_CUBIC)
  return lp


'''
def detectContours(lp): # returns an array of left to right filterd contours positions (x,y,w,h)
  org = lp.copy()  
  lp = lp.astype(np.uint8)
  #_, contours, _ = cv2.findContours(lp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  major = cv2.__version__.split('.')[0]
  if major == '3':
    _, contours, _ = cv2.findContours(lp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    contours, _ = cv2.findContours(lp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



  if (show_contours):
    cv2.drawContours(org, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('all contours', org)
    cv2.waitKey(0)
  
  img_area = imageArea(lp.shape)
  charBoxesArray = []
  for index, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    roi_area = w*h
    roi_ratio = roi_area/img_area
    #if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
    #proportion = (w*h)/ area
    #if (proportion > 0.3 or proportion < 0.012 or x == 0 or y == 0 ): # filter big and small contours
    #  continue
    if (roi_ratio < 0.015 or roi_ratio > 0.09):
      continue
    #charBoxesArray.append(adjustSize(x,y,w,h,hOrg))
    charBoxesArray.append([x,y,w,h])
    if show_characters:
      mask = lp[y:y+h,x:x+w]
      #cv2.drawContours(mask, [cnt], -1, (0,255,0), 1)
      cv2.imshow('contour %d'%(index) ,mask)
  if show_characters:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  charBoxesArray = sorted(charBoxesArray, key = lambda p: p[0]) # sort from left to right
  #print ('the contours: ', charBoxesArray)

  return np.array(charBoxesArray)
'''


def detectContours(lp, org): # returns an array of left to right filterd contours positions (x,y,w,h)
  #org = lp.copy()  
  lp = lp.astype(np.uint8)
  major = cv2.__version__.split('.')[0]
  if major == '3':
    _, contours, _ = cv2.findContours(lp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  else:
    contours, _ = cv2.findContours(lp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  if (show_contours):
    cv2.drawContours(org, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('all contours', org)
    cv2.waitKey(0)
  
  img_area = imageArea(lp.shape)
  charBoxesArray = []
  for index, cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)
    roi_area = w*h
    roi_ratio = roi_area/img_area
    #if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
    #proportion = (w*h)/ area
    #if (proportion > 0.3 or proportion < 0.012 or x == 0 or y == 0 ): # filter big and small contours
    #  continue
    if (roi_ratio < 0.015 or roi_ratio > 0.09):
      continue
    #charBoxesArray.append(adjustSize(x,y,w,h,hOrg))
    charBoxesArray.append([x,y,w,h])
    if show_characters:
      mask = lp[y:y+h,x:x+w]
      ch = org[y:y+h,x:x+w]
      cv2.drawContours(mask, [cnt], -1, (0,255,0), 1)
      plt.imshow(mask)
      plt.show()
      print (predictChar(charModel, ch))
  # if show_characters:
  #   cv2.waitKey(0)
  #   cv2.destroyAllWindows()
  charBoxesArray = sorted(charBoxesArray, key = lambda p: p[0]) # sort from left to right
  #print ('the contours: ', charBoxesArray)

  return np.array(charBoxesArray)



def imageArea(shp):
  return shp[0]*shp[1]

def carIndex(name):
  index = int(re.findall(r'_(.*?)\.',name)[0])
  return index

from PlateExtractionSeg import createUnetModel, predictLP

'''def checkCE(model):
    path = PATH + 'original_images/'
    for index, filename in enumerate(sorted(os.listdir(path), key=carIndex)):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
          img = plt.imread(path + filename)
          lp = predictLP(model, img)
          detectContours(lp)
          cv2.waitKey(0)

    
    cv2.destroyAllWindows()'''
    
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    # return the edged image
    return edged

def something(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
  edges = auto_canny(thresh_inv)
  ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
  img_area = img.shape[0]*img.shape[1]
  for i, ctr in enumerate(sorted_ctrs):
      x, y, w, h = cv2.boundingRect(ctr)
      roi_area = w*h
      roi_ratio = roi_area/img_area
  if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
          if ((h>1.2*w) and (3*w>=h)):
              cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)


# model = createUnetModel(None,None,False)
# model.load_weights(PATH + 'models/model_12ep_weights.h5')
# print ('Model loaded.')
