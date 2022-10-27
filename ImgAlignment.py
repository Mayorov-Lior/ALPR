import cv2
import numpy as np
from ImgUtilities import *
import imutils
 
PATH = 'C:/Users/Lior/Documents/work/License Plate Recognition/'
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

check = False
show_details = False
 
'''
def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
  
def mainAlignImg(im):
   
    # Read reference image
    refFilename = PATH + 'lp_1283.jpg'
    #refFilename = PATH + 'image-registration-example-ref.jpg'
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    imReference = fitRegImgSize(im, imReference)
    cv2.imshow('ref', imReference)

    print("Aligning images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk. 
    #   outFilename = "aligned.jpg"
    #   print("Saving aligned image : ", outFilename); 
    #   cv2.imwrite(outFilename, imReg)
    cv2.imshow('aligned', imReg)
    cv2.waitKey(0)
    # Print estimated homography
    print("Estimated homography : \n",  h)

def fitRegImgSize(imgToAlign, imgRef):
    hA,wA = imgToAlign.shape[:2]
    hR,wR = imgRef.shape[:2]
    print (hA,wA)
    print (hR,wR)

    newH = int((hA/wA)*wR)
    add = abs(int((newH - hR)/2))
    imgRef = cv2.copyMakeBorder(imgRef,add,add,0,0,cv2.BORDER_CONSTANT,value=0)
    imgRef = cv2.resize(imgRef, (hA,wA)) 
    return imgRef
   
def mainAlignImg (img1_color):
    img2_color = cv2.imread(PATH + 'lp_1283.jpg')    # Reference image. 
    #img2_color = cv2.imread(PATH + 'image-registration-example-ref.jpg')    # Reference image. 
    img2_color = fitRegImgSize(img1_color, img2_color)
    cv2.imshow('ref', img2_color)
    
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 
    
    orb_detector = cv2.ORB_create(5000) 
    
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
    matches = matcher.match(d1, d2) 
    matches.sort(key = lambda x: x.distance) 
    
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 
    
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 
    
    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 
    
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
    transformed_img = cv2.warpPerspective(img1_color, homography, (width, height)) 
    cv2.imshow('aligned', transformed_img)
    cv2.waitKey(0)
'''

'''def getLpRotationInfo(lp):
    firstContour = sorted(getContours(lp), key=cv2.contourArea, reverse=True)[0]
    lpRotatedBox = cv2.minAreaRect(firstContour) # top-left corner(x,y), (width, height), angle of rotation
    #lpPoints = cv2.convexHull(firstContour)
    
    return lpRotatedBox'''

def showTilted(lpCpy, lpRotatedBox, name):
    box = cv2.boxPoints(lpRotatedBox)
    box = np.int0(box)
    cv2.drawContours(lpCpy, [box], 0, (0,255,0), 3)
    cv2.imshow(name, lpCpy) 

'''def alignImg (lp):
    lpRotatedBox1 = getLpRotationInfo(lp)
    angle = lpRotatedBox1[2]

    lpRotated = imutils.rotate_bound(lp, -angle)  
    if show_details:
        showTilted(lp.copy(), lpRotatedBox1, 'first rotation analysis')

    #lpRotatedBox2 = getLpRotationInfo(lpRotated)
    #(w,h) = lpRotatedBox2[1]
    [h,w] = lpRotated.shape[:2]

    if (h > w):
        if show_details:
            pass
            #showTilted(lpRotated.copy(), lpRotatedBox2, 'wrong rotated analysis')
        lpRotated = imutils.rotate_bound(lpRotated, 90)  

    if show_details:
        cv2.imshow('rotated', lpRotated) 
        print ('===================', angle, '===================')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # secondContour = sorted(getContours(lpRotated), key=cv2.contourArea, reverse=True)[1]
    # lpRotatedBox = lpRotated.copy()
    # cv2.drawContours(lpRotatedBox, [secondContour], 0, (0,255,0), 3)
    # cv2.imshow('3', lpRotatedBox) 

    # lpBox = cv2.boundingRect(secondContour)
    # cv2.imshow('4', lpBox) 

    return lpRotated'''

#######################################################################################################
#@title Align image 2

def getLargestCntFromBW(thImg):
  cnts = sorted(getContoursFromTh(thImg), key=cv2.contourArea, reverse=True)
  if len(cnts) == 0:
      return []
  return cnts[0]

def getLpRotationInfo(lp):
    h, w = lp.shape[:2]
    imgArea = h * w
    largest_cnt = 42
    if len(lp.shape) == 3: # bgr
      imgGrey = cv2.cvtColor(lp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
      th_opt = [127,118,135,100,85,150,70,180]
      for th in th_opt:
          _, thImg = cv2.threshold(imgGrey, th, 255, cv2.THRESH_BINARY)
          plt.imshow(thImg)
          plt.show()
          largest_cnt =  getLargestCntFromBW(thImg)
          if len(largest_cnt) == 0:
            continue
          [_,_,wBB,hBB] = cv2.boundingRect(largest_cnt)
          if (wBB == w and hBB == h):
              continue
          if (((wBB/w) > 0.6) and ((hBB/h) > 0.6)):
              break
    else: # bw lp
      largest_cnt =  getLargestCntFromBW(lp)
    
    if not isinstance(largest_cnt, np.ndarray):
        print ('error in rotation: largest_cnt = ', largest_cnt)
        return False, False
    
    lpRotatedBox = cv2.minAreaRect(largest_cnt) # top-left corner(x,y), (width, height), angle of rotation    
    return lpRotatedBox, largest_cnt

def rotate_bound(image, angle):
    print ('yo')
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]
    
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.WARP_FILL_OUTLIERS)

def fixLPDirection(lp, model): # gets lp image with vertical plate, and rotates it back to 0 degrees 
  lp_LP_SIZE = resizeImgToSquare(lp, LP_SIZE)
  rotation = model.predict(np.array([lp_LP_SIZE]))[0]
  rotation = rotation.argmax()

  # lp_pil = Image.fromarray(lp)
  # lp_fixed = np.array(lp_pil.rotate(90*(-rotation), resample=Image.BICUBIC, expand=True))
  lp_fixed = rotate_bound(lp, 90*(-rotation))
  return lp_fixed

def alignImg (lp, mask, lpModel, alignModel):
  lpRotatedBox1, cnt = getLpRotationInfo(mask) # top-left corner(x,y), (width, height), angle of rotation
  if lpRotatedBox1 == False: # no contours
      return lp, 0
  angle = lpRotatedBox1[2]
  angle = -angle
  
  if (lp.max() <= 1):
    lp = (lp*255).astype(np.uint8)
  # lp_pil = Image.fromarray(lp)
  # lpRotated = np.array(lp_pil.rotate(-angle, resample=Image.BICUBIC, expand=True))

  if angle > 10:
    lpRotated = rotate_bound(lp, angle)  
  # now we have a vertical plate- here we fix the direction with a cnn model
  lpRotated = fixLPDirection(lpRotated, alignModel)

  return lpRotated, angle






if check:
    
    import os
    for filename in sorted(os.listdir(PATH + 'lp_images/'), key=getCarIndex)[880:]:
        #fileToRead = PATH + 'lp_images/lp_746.jpg'
        fileToRead = PATH + 'lp_images/' +  filename
        lp = cv2.imread(fileToRead , cv2.IMREAD_COLOR)
        alignImg (lp)
        #img1_color = cv2.imread(PATH + 'image-registration-example.jpg' , cv2.IMREAD_COLOR)
        #cv2.imshow('org2', lp) 
        #cv2.waitKey(0)
