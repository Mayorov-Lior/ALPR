import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import imutils
#import matplotlib

PATH = 'C:/Users/Lior/Documents/work/License Plate Recognition/'
CATAGORIES_NUM = 12 # 0,1,2,3,4,5,6,7,8,9,צ,*(nothing)
IMG_SIZE = 224

def getCarIndex(name):
  index = int(re.findall(r'_(.*?)\.',name)[0])
  return index

def CropImgToSquare(img): # crops the image by the smaller egde to the centered square
  h,w = img.shape[:-1]
  centerX = w/2
  centerY = h/2
  if (w > h):
    cropped = img[:, int(centerX - centerY): int(centerX + centerY)]
  else:
    cropped = img[int(centerY - centerX): int(centerY + centerX),:]
  return cropped

def fitImgSize(img, edge): # crops to square and resizes
  filled = getDevidableImg(img)
  cropped = CropImgToSquare(filled)
  resized = cv2.resize(cropped, (edge, edge), interpolation=cv2.INTER_AREA) 
  return resized

def getDevidableImg(img): # get devidable edge by IMG_SIZE
  getDevidableEdge = lambda e: IMG_SIZE * int(e / IMG_SIZE) 
  h, w = img.shape[:-1]
  add = abs (min(h,w) - min(getDevidableEdge(h), getDevidableEdge(w)))
  filled = fillSides(img, add)
  return filled

def fillSides(img, add): # padd sides with zeros
  img = img.astype(np.uint8)
  h, w = img.shape[:-1]
  add1 = int(add/2.0)
  add2 = add - add1
  if (w > h):
    filled = cv2.copyMakeBorder(img,add1,add2,0,0,cv2.BORDER_CONSTANT,value=0)
  else:
    filled = cv2.copyMakeBorder(img,0,0,add1,add2,cv2.BORDER_CONSTANT,value=0)
  return filled

def resizeImgToSquare(img, edge): # fills image to square by the smaller egde, and resizes
  h,w = img.shape[:-1]
  diff = abs(w - h)
  filled = fillSides(img, diff)
  resized = cv2.resize(filled, (edge, edge)) 
  return resized

def getLPbyAdjustingBox(box, img): # recieves box by the fitted 224x224 image and returns cropped lp from org image
  # img = CropImgToSquare(img)
  # edgeOrg = img.shape[0] # img is a square
  # [x,y,w,h] = [int((elem/IMG_SIZE)*edgeOrg) for elem in box]
  # lp = img[y:y+h, x:x+w]

  return resizeImgToSquare(lp, IMG_SIZE)

def imagesToNPArray(imgsFolder, name, start = 0):
  arr = 0
  progress = True
  if (start != 0):
    arr = list(np.load(PATH + name))
  path = PATH + imgsFolder
  for index, filename in enumerate(sorted(os.listdir(path), key=getCarIndex)[start:]):
      if filename.endswith(".jpg") or filename.endswith(".png"):
        if (progress and (index % 50 == 0)):
          print (filename + ' was saved in array.')
        img = plt.imread(path + filename)
        arr.append(img)
  arr = np.stack( arr, axis=0 )
  np.save(PATH + name, arr)
  return arr

def getContoursFromTh(img):
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours
    
def getContours(img):
  imgGrey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  _, th = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)
  return getContoursFromTh(th)

def blackoutBackground(lp, cnt): # masking to black out background- everything except the contour
  lpCropped = lp.copy()
  mask = np.zeros(lp.shape)
  cv2.drawContours(mask, [cnt], -1, [255,255,255], thickness = -1)
  lpCropped[mask == 0] = mask[mask == 0]
  return lpCropped

def blackoutBackgroundFromMinAreaRect(img, imgRotatedBox): # masking to black out background- everything except the contour
  # top-left corner(x,y), (width, height), angle of rotation
  imgCropped = img.copy()
  mask = np.zeros(img.shape)

  box = cv2.boxPoints(imgRotatedBox)
  box = np.int0(box)
  cv2.drawContours(mask, [box], -1, [255,255,255], thickness = -1)
  [x,y,w,h] = cv2.boundingRect(box)
   
  imgCropped[mask == 0] = mask[mask == 0]
  imgCropped = imgCropped[y:y+h, x:x+w]

  return imgCropped




def fragmentOfBlackPixels(img):
  img =  img.astype('uint8')
  h, w = img.shape[:2]
  numOfPixels = h * w
  numOfNonBlackPixels = cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
  fragment = (numOfPixels  - numOfNonBlackPixels)/(numOfPixels) 
  return fragment 