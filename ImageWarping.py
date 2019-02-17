# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID: 5292195
"""
"""
reference taken from 
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html,
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html,
https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545
"""

import cv2
import numpy as np

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    cv2.imwrite(name,img)
    #print("\n****" + name + " written to disk****")
    #print("height:",len(img))
    #print("width:",len(img[0]))
    
minkeyCount = 4
        
mountain1 = cv2.imread("mountain1.jpg",0)
mountain2 = cv2.imread("mountain2.jpg",0)
mountain1color = cv2.imread("mountain1.jpg",1)
mountain2color = cv2.imread("mountain2.jpg",1)

mount1 = mountain1.copy()
mount2 = mountain2.copy()
mount1color = mountain1color.copy()
mount2color = mountain2color.copy()

"""create sift object"""
sift = cv2.xfeatures2d.SIFT_create()

"""detect and compute sift keypoints and descriptors"""
keypt_1, descp_1 = sift.detectAndCompute(mountain1,None)
keypt_2, descp_2 = sift.detectAndCompute(mountain2,None)

"""draw all the keypoints on the colored images"""
mt1=cv2.drawKeypoints(mount1color,keypt_1,mount1color)  #drawKeypoints(sourceimage,keypoints,outputimage)
mt2=cv2.drawKeypoints(mount2color,keypt_2,mount2color)

writeImage("ImageWarping_sift1.jpg",mt1)
writeImage("ImageWarping_sift2.jpg",mt2)

"""The Brute Force matcher is used for feature matching between the two images."""
bf_matcher = cv2.BFMatcher()

"""BFMatcher.knnMatch() function is used to find 2 best matches for each Keypoint."""
calcmatches = bf_matcher.knnMatch(descp_1,descp_2, k=2)
validMs = []
"""calculate the matches that pass the ratio test with threshold as 0.75"""
for m,n in calcmatches:
    if m.distance < 0.75*n.distance:
        validMs.append(m)

"""matches are drawn for all the valid matches"""
matchedImgs = cv2.drawMatches(mountain1color,keypt_1,mountain2color,keypt_2,validMs,None,flags=2)
writeImage("ImageWarping_matches_knn.jpg",matchedImgs)


if len(validMs) > minkeyCount:
    """keypoints in the first image are calculated"""
    sourcePts = np.float32([keypt_1[m.queryIdx].pt for m in validMs])
    """keypoints in the 2nd image are calculated"""
    destPts = np.float32([keypt_2[n.trainIdx].pt for n in validMs])
    """homography and mask is calculated using RANSAC algorithm"""
    Homography, mask = cv2.findHomography(sourcePts, destPts, cv2.RANSAC,5.0)
    mask = np.ndarray.flatten(mask).tolist()
    print("\nHomography Matrix: \n",Homography)
    h1,w1 = mountain1.shape
    h2,w2 = mountain2.shape
    """
    corner points of the images are taken to calculate the necessary orientation 
    change
    """
    pt1 = np.float32([[[0,0],[0,h1],[w1,h1],[w1,0]]])
    pt2 = np.float32([[[0,0],[0,h2],[w2,h2],[w2,0]]])
    
    pTrans = cv2.perspectiveTransform(pt1,Homography)
    
    conPts = np.concatenate((pTrans.reshape(4,2),pt2.reshape(4,2)),axis=0)
    [x_min, y_min] = np.int32(np.amin(conPts,axis=0))
    [x_max, y_max] = np.int32(np.amax(conPts,axis=0))
    transmin = [-x_min,-y_min]
    transmax = [x_max,y_max]
    fmask = np.zeros((len(mask)))
    inlierIndxs = np.where(np.array(mask) == 1)[0]
    np.random.shuffle(inlierIndxs)
    inlierIndxs = inlierIndxs[:10]
    fmask[inlierIndxs] = 1
    """
    the translation is computed that needs to be applied on the first image so that
    it doesn't get cropped
    """
    Htrans = [[1,0,transmin[0]],[0,1,transmin[1]],[0,0,1]]
    persMatchImg = cv2.drawMatches(mountain1color,keypt_1,mountain2color,keypt_2,validMs,None,matchColor = (255,0,0),matchesMask = fmask,flags = 2)
    writeImage("ImageWarping_matches.jpg",persMatchImg)
    mount1color = mountain1color.copy()
    mount2color = mountain2color.copy()
    
    wrapImage = cv2.warpPerspective(mount1color, np.dot(Htrans,Homography), (transmax[0]+transmin[0],transmax[1]+transmin[1]))
    wrapImage[transmin[1]:h2+transmin[1],transmin[0]:w2+transmin[0]] = mount2color
    writeImage("ImageWarping_pano.jpg",wrapImage)
else:
    print("Not enough keypoints to find the Homography. Should be at least ",minkeyCount)
    #mask = None



