
#!unzip -q /content/drive/"My Drive"/lp_data/images.zip -d /content/drive/"My Drive"/lp_data/images
#!pip install patool
#import patoolib
#patoolib.extract_archive("/content/drive/My Drive/lp_data/images.rar", outdir="/content/drive/My Drive/lp_data/images")

# Commented out IPython magic to ensure Python compatibility.
import cv2
import math
import numpy as np
import os
import sys
from json import JSONDecoder, JSONDecodeError
from ImgUtilities import resizeImgToSquare, fillSides
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tqdm import tqdm_notebook 
import matplotlib.pyplot as plt
import matplotlib
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from sklearn.metrics import jaccard_score
from keras import regularizers
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
#from keras.optimizers import Adam, SGD
#from keras.models import Sequential, load_model, Model
#from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input, Dense, Dropout, Flatten, UpSampling2D
from keras import backend

# %matplotlib inline

'''def countFiles(path):
  count = 0
  for file in os.listdir(path):
    count += 1
  return count

count = countFiles('/content/drive/My Drive/lp_data/LPimagesBW/')
print ('number of files: ', count)'''

# https://github.com/jrieke/shape-detection/blob/master/single-rectangle.ipynb
# unet model- https://github.com/zhixuhao/unet/blob/master/model.py
# FCN model - https://github.com/aurora95/Keras-FCN/blob/master/models.py
PATH = '/content/drive/My Drive/lp_data/'
NOT_WHITESPACE = re.compile(r'[^\s]')
NUM_IMGS = 665
IMG_SIZE = 224

# losses
# ref: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
     
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(-1,-2,-3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(-1,-2,-3))

  return 1 - numerator / denominator
  
def splitData(data, percent):
  i = int(percent * data.shape[0]) # splitting
  train = data[:i]
  test = data[i:]
  return train, test

def prepareImgs(imgs, percent):
  X = imgs
  if (imgs.max() > 1):
    X = imgs / 255.0
  return splitData(X, percent)

class CustomSaver(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (epoch % 1 == 0):
      self.model.save_weights(PATH + "model_{}.hd5".format(epoch))
      print ('   saved after %d epochs.'%(epoch))
  
def UnetOutput(inputs):
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)
  
  '''vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_size)

  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(y.shape[-1], activation='sigmoid'))

  #for l in vgg.layers:
  #  l.trainable = False'''
  
  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  
  #conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  conv10 = Reshape((224,224))(conv10)
  return conv10

def createUnetModel(train_X, train_y, fit=True):
  input_size = (224,224,3)
  epochs_num = 12
  
  inputs = Input(input_size)
  outputs = UnetOutput(inputs)
  model = Model(input = [inputs], output = [outputs])

  adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
  
  #model.compile(optimizer = adam, loss = dice_coef_loss, metrics = ['accuracy'])
  model.compile(optimizer = adam, loss = dice_loss, metrics = ['accuracy'])
  
  model.summary()
  saver = CustomSaver()
  if fit:
    model.fit(train_X, train_y, callbacks=[saver], epochs=epochs_num, validation_split=0.2, verbose=1)
  return model

def predictLPImgs(model, imgs):
  pred = model.predict(imgs)
  pred = (pred - pred.min())/(pred.max()-pred.min())
  return pred

def predictAdjustedLP(model, img):
  X = np.array([img])
  lp = predictLPImgs(model, X)[0]
  lp = applyMask(img, lp, False)
  return lp

def predictLP(model, img):
  if img.max() <= 1:
      img = img/255.0
  hOrg, wOrg = img.shape[:2]
  maxEdge = max(hOrg, wOrg)
  imgResized = resizeImgToSquare(img, IMG_SIZE)
  lp = predictLPImgs(model, np.array([imgResized]))[0]

  x,y,w,h = getBox(lp.astype(np.uint8))
  lp = imgResized[y:y+h,x:x+w]

  x = int((x/IMG_SIZE)*maxEdge)
  w = int((w/IMG_SIZE)*maxEdge)
  y = int((y/IMG_SIZE)*maxEdge)
  h = int((h/IMG_SIZE)*maxEdge)

  imgPadded = fillSides(img, abs(wOrg - hOrg))
  lp = imgPadded[y:y+h,x:x+w]
  return lp

def applyMask(img, mask, show):
  mask = mask.astype(np.uint8)
  img = cv2.bitwise_and(img,img,mask = mask)
  x,y,w,h = getBox(mask)
  lp = img[y:y+h,x:x+w]
  if (show):
    plt.imshow(lp)
    plt.show()
  return lp

def getBox(mask):
  #_, th = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),50,255,cv2.THRESH_BINARY)
  #th = cv2.erode(mask, np.ones(2), iterations=1) 
  major = cv2.__version__.split('.')[0]
  if major == '3':
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #(_, contours, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key = cv2.contourArea, reverse = True)
  if (len(contours) == 0):
    h,w = mask.shape
    return [0,0,w,h]

  contour = contours[0]
  #cv2.drawContours(img, [contour], -1, (0,255,0), 3)
  x,y,w,h = cv2.boundingRect(contour)
  return x,y,w,h

def showResults(test_X, test_y, pred, limit=5):
  mask = 1
  if mask:
    for i in range(limit):
      applyMask(test_X[i], pred[i], True)
  else:
    for i in range(limit):
      plt.imshow(test_X[i])
      plt.show()

      plt.imshow(test_y[i])
      plt.show()

      plt.imshow(pred[i])
      plt.show()
      #print (pred[i].max())

def dice(p,truth):
  k = 1
  return (np.sum(p[truth==k])*2.0) / (np.sum(p) + np.sum(truth))

def avgDice(pred_seg, test_seg):
  # Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset. 
  noOverlap = 0
  avg = 0.0
  for pred_bbox, test_bbox in zip(pred_seg, test_seg):
    truth = test_bbox.flatten()
    pred = pred_bbox.flatten()
    dice = dice(pred, truth)
    avg += dice
    if (dice == 0):
      noOverlap += 1
  avg = avg / len(pred_seg)
  return avg, noOverlap

def extractPlate(imgs, seg):    
  percent = 0.8
  train_X, test_X = prepareImgs(imgs, percent)
  train_y, test_y = prepareImgs(seg, percent)
  print ('imgs proccessed')
  model = createUnetModel(train_X, train_y,True)
  model.save_weights(PATH + 'my_model_weights.h5')
  pred = model.predict(test_X)
  pred = (pred - pred.min())/(pred.max()-pred.min())

  dice, noOverlap = avgDice(pred, test_y)
  print ('avg dice: ', dice, '; no overlaps: ', noOverlap)

  showResults(test_X, test_y, pred, 20)

  return model, imgs, seg, pred

def operateModel(name):
  imgs = np.load(PATH + 'imgsArray.npy')
  seg = np.load(PATH + 'bwArray.npy')
  
  model = createUnetModel(None,None,False)
  model.load_weights(name)
  print ('Model loaded.')
  
  _, test_X = prepareImgs(imgs, 0.8)
  pred = model.predict(test_X)
  _, test_y = prepareImgs(seg, 0.8)
  dice, noOverlap = avgDice(pred, test_y)
  print ('avg dice: ', dice, '; no overlaps: ', noOverlap)
  showResults(test_X, test_y, pred, 20)
  
#operateModel(PATH + 'my_model_weights.h5')

#imgs = np.load(PATH + 'imgsArray.npy')
#seg = np.load(PATH + 'bwArray.npy')
#print ('Extracting license plate...')
#model, imgs, seg, pred = extractPlate(imgs, seg)