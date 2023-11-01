# import cv2
# import numpy as np
# import os
# import re
# import matplotlib.pyplot as plt
# import imutils

import os
import re
import cv2
from PIL import Image
import numpy as np
import sys
from collections import Counter
lp_path = "./content/ALPR/"
if lp_path not in sys.path:
  sys.path.append(lp_path)
from ImgUtilities import *
# from ImgUtilities import getContoursFromTh
from PlateExtractionSeg import *
# from PlateExtractionSeg import dice_loss, showResults
from CharactersExtraction import *
# from CharactersExtraction import detectContours
from CharClassification import *
import matplotlib.pyplot as plt
import time
# %matplotlib inline

x = 2

# Permenant Parameters
IMG_SIZE = 224
LP_SIZE = 100
CHAR_IMG_SIZE = 28
CATAGORIES_NUM = 12 # 0,1,2,3,4,5,6,7,8,9,m
# createUnetModel, predictLP, splitData, prepareImgs, predictAdjustedLP, predictLPImgs,

def classifyLps(charModel, lps, org): # recieves list of lps images, returns list of identified characters positions (positionsList) and a list of lists of predicted chars for each lp (numbersList)
  numbersList = []
  positionsList = []
  for l in lps:
    contours = detectContours(l, org)
    numbersL, positionsL = charImgsToStr(charModel, org, contours)

    if len(numbersL) > 0 :
      numbersList.append(numbersL)
      positionsList.append(positionsL)
  return positionsList, numbersList

def findNumbersOrder(finalNumber): # need to finish this
  finalNumber = sorted(finalNumber, key = lambda p: p[1][0]) # sort from left to right
  return finalNumber


def findFinalNumberFromLps(positionsList, numbersList):
  finalNumber = [] # [ [char, pos],  ... ]
  while (len(positionsList) > 0): # iterated all lists
    positionsList, numbersList, finalNumber = iterateFirstOption(positionsList, numbersList, finalNumber)
  finalNumber = findNumbersOrder(finalNumber)
  return finalNumber

def iterateFirstOption(positionsList, numbersList, finalNumber):
  while (len(positionsList[0]) > 0): # iterate all first list's elements
    chars = [] # all chars predicted at current position
    elementToCheck = positionsList[0][0]
    chars.append(numbersList[0][0])
    del positionsList[0][0]
    del numbersList[0][0]

    positionsList, numbersList, chars = iterateListsForElementToCheck(positionsList, numbersList, chars,elementToCheck)
    # if len(chars) <= 2:
    #   continue
    counter = Counter(chars)
    maxVoteChar = max(counter, key=counter.get)
    finalNumber.append([maxVoteChar, elementToCheck])

  del positionsList[0]
  del numbersList[0]
  return positionsList, numbersList, finalNumber

def iterateListsForElementToCheck(positionsList, numbersList, chars,elementToCheck):
  numberOfLists = len(positionsList)
  for index in range(1,numberOfLists):
    for indexInList in range(len(positionsList[index])):
      if isSameChar(elementToCheck, positionsList[index][indexInList]):
        chars.append(numbersList[index][indexInList])
        del positionsList[index][indexInList]
        del numbersList[index][indexInList]
        break
  return positionsList, numbersList, chars

def findCharacters(charModel, lp):
  if lp.max() <= 1:
    lp = (lp*255).astype('float32')
  org = resizeLP(lp)
  lps = proccessLP(lp) # all sorts of processing and thresholding of lp

  # positionsList, numbersList = classifyLps(charModel, lps, org)

  # positionsList2, numbersList2 = classifyLps(charModel2, lps, org)
  # positionsList = positionsList + positionsList2
  # numbersList = numbersList + numbersList2

  positionsList, numbersList = classifyLps(charModel, lps, org)
  numbers, positions = findCharactersFromLists(positionsList, numbersList)
  return numbers, positions


def findCharactersFromLists(positionsList, numbersList):
  finalNumber = findFinalNumberFromLps(positionsList, numbersList)
  numbers = ''.join([elem[0] for elem in finalNumber])
  positions = [elem[1] for elem in finalNumber]

  # print ('numbers: ', numbers)
  return numbers, positions

def findLPNumber(charModel, lp):
  return findCharacters(charModel, lp)[0]

def findLPPositions(charModel, lp):
  return findCharacters(charModel, lp)[1]

def isValidLPLength(arr):
  return (len(arr) == 7 or len(arr) == 8)

def charImgsToStr(charModel, lp, arr):
  validChars = ['0','1','2','3','4','5','6','7','8','9','m']
  charArray = []
  positionsArray = []
  for [x,y,w,h] in arr:
    charImg = lp[y:y+h, x:x+w]
    ch = predictChar(charModel, charImg)

    if ch in validChars:
      # print ('predicted char: ', ch)
      # plt.imshow(charImg)
      # plt.show()
      if ch == 'm':
        charArray.append('צ')
      else:
        charArray.append(ch)
      positionsArray.append([x,y,w,h])
  return charArray, positionsArray


def predictLP(model, img, alignModel):
  # if img.max() > 1:
  #     img = img/255.0
  hOrg, wOrg = img.shape[:2]

  # print ('org: ')
  # s = 10
  # plt.figure(figsize = (s,s))
  # plt.imshow(img/255.0)
  # plt.show()

  # resizing img for lp detection model
  imgResized = resizeImgToSquare(img.copy(), IMG_SIZE)
  mask_IMG_SIZE = predictLPImgs(model, np.array([imgResized]))[0].astype(np.uint8)

  # padding org image to get a square image and adjusting mask
  imgPadded = fillSides(img, abs(wOrg - hOrg))
  print ('imgPadded and org equal: ', np.array_equal(imgPadded, img))
  mask_ORG_SIZE = cv2.resize(mask_IMG_SIZE, imgPadded.shape[:2], interpolation = cv2.INTER_AREA)

  # masking image
  cnts = sorted(getContoursFromTh(mask_ORG_SIZE), key=cv2.contourArea, reverse=True)
  if len(cnts) == 0:
      return img
  largest_cnt = cnts[0]
  rect = cv2.minAreaRect(largest_cnt) # top-left corner(x,y), (width, height), angle of rotation
  points = cv2.boxPoints(rect).astype(np.int)

  smoothed_mask = np.zeros(imgPadded.shape)
  cv2.fillPoly(smoothed_mask, pts = [points], color=(1,1,1)) # smoother mask

  # copy for later rotation
  smoothed_mask_cpy = cv2.cvtColor(smoothed_mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)

  # replace white with the lp
  indices = (smoothed_mask[:] == 1)
  smoothed_mask[indices] = imgPadded[indices]

  # print ('smoothed_mask: ', smoothed_mask.shape)
  # plt.figure(figsize = (s,s))
  # plt.imshow(smoothed_mask/255.0)
  # plt.show()

  x,y,w,h = cv2.boundingRect(largest_cnt) # to crop
  lp = (smoothed_mask[y:y+h, x:x+w])
  # print ('cropped smoothed_mask: ', lp.shape)
  # plt.figure(figsize = (s,s))
  # plt.imshow(lp/255.0)
  # plt.show()

  if lp.max() > 1:
    lp = lp/255.0
  #lp, angle = alignImg ((lp/255.0).copy(), smoothed_mask_cpy, model, alignModel)


  return lp, points

def rearrangeNumber(num):
  if 'צ' not in num:
    if (len(num) == 7):
      num = num[:2] + '-' + num[2:5] + '-' + num[5:]
    elif (len(num) == 8):
      num = num[:3] + '-' + num[3:5] + '-' + num[5:]
  else:
    num = num[:-1] + '-' + num[-1]
  return num

def showResult(img, points, num):
  cv2.drawContours(img, [points], 0, (0,255,0), 2)
  x,y,w,h = cv2.boundingRect(points)
  size = 12
  xText = x
  yText = y - 10
  if yText < 0:
    yText = y + h + 10
  s = str(num)
  plt.text(xText, yText, s, size=size, bbox=dict(fill=True, facecolor='red', alpha=0.5))
  plt.imshow(img)
  plt.show()

def findNumberFromCarImg(img, lpModel, AlignModel, charModel):
  if type(img)==str:
    img = plt.imread(img)
  lp, points = predictLP(lpModel, img.copy(), AlignModel) # get lp image
  num = findLPNumber(charModel, lp)
  num = rearrangeNumber(num)
  return num, points

def getCarIndex(name):
  index = int(re.findall(r'_(.*?)\.',name)[0])
  return index

# PATH = '/content/drive/My Drive/lp_data/'
# path = PATH + 'car_images/'


# for filename in sorted(os.listdir(path), key = getCarIndex)[:40]:
#   print ('======================== car ', filename, '========================')
#   car = plt.imread(path + filename)
#   num, points = findNumberFromCarImg(car, lpModel, AlignModel, charModel)
#   showResult(car, points, num)




'''
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
  '''