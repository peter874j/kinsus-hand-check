import cv2
import numpy as np
import os


def main():

    ### PCB (99, 55), (841, 538)
    ### GlovesL (1, 494), (131, 619)
    ### GlovesR (797, 498), (942, 621)
    # image = cv2.imread("perspective_29.jpg")
    image = cv2.imread("perspective_23.jpg")
    (imgH, imgW, _) = image.shape
    pixelThres = 30
    ### Point of PCB、GlovesL、GlovesR
    '''
    PCBPoint = [[99, 55], [841, 538]]
    GlovesLPoint = [[1, 494], [131, 619]]
    GlovesRPoint = [[797, 498], [942, 621]]
    '''
    PCBPoint = [[99, 55], [834, 526]]
    GlovesLPoint = [[7, 253], [145, 421]]
    GlovesRPoint = [[784, 238], [940, 429]]
    ### Region of GlovesL、GlovesR
    GlovesLImg = image[GlovesLPoint[0][1]:GlovesLPoint[1][1], GlovesLPoint[0][0]:GlovesLPoint[1][0]]
    GlovesRImg = image[GlovesRPoint[0][1]:GlovesRPoint[1][1], GlovesRPoint[0][0]:GlovesRPoint[1][0]]
    ### Mask of PCB、Gloves
    zeroMask = np.zeros((imgH, imgW), np.uint8)
    resultsMask = np.zeros((imgH, imgW), np.uint8)
    PCBMask = np.zeros((imgH, imgW), np.uint8)
    glovesMask = np.zeros((imgH, imgW), np.uint8)
    PCBMask[PCBPoint[0][1]:PCBPoint[1][1], PCBPoint[0][0]:PCBPoint[1][0]] = 255
    ### Mask of GlovesL、GlovesR
    lowerGloves = np.array([210, 210, 210])
    upperGloves = np.array([255, 255, 255]) 
    maskGlovesL = cv2.inRange(GlovesLImg, lowerGloves, upperGloves)
    maskGlovesR = cv2.inRange(GlovesRImg, lowerGloves, upperGloves)
    ### Open Operation
    kernel = np.ones((5, 5),np.uint8)
    maskGlovesL = cv2.morphologyEx(maskGlovesL, cv2.MORPH_OPEN, kernel)
    maskGlovesR = cv2.morphologyEx(maskGlovesR, cv2.MORPH_OPEN, kernel)
    ### Paste to Gloves Mask
    glovesMask[GlovesLPoint[0][1]:GlovesLPoint[1][1], GlovesLPoint[0][0]:GlovesLPoint[1][0]] = maskGlovesL
    glovesMask[GlovesRPoint[0][1]:GlovesRPoint[1][1], GlovesRPoint[0][0]:GlovesRPoint[1][0]] = maskGlovesR
    ### Bitwise PCB&Gloves Mask
    resultsMask = cv2.bitwise_and(PCBMask, glovesMask)
    thresMask = resultsMask[PCBPoint[0][1] + pixelThres:PCBPoint[1][1] - pixelThres, PCBPoint[0][0] + pixelThres:PCBPoint[1][0] - pixelThres]
    zeroMask[PCBPoint[0][1] + pixelThres:PCBPoint[1][1] - pixelThres, PCBPoint[0][0] + pixelThres:PCBPoint[1][0] - pixelThres] = thresMask
    cv2.imshow('PCBMask', PCBMask)
    cv2.imwrite('PCBMask.jpg', PCBMask)
    cv2.waitKey(0)   
    cv2.imshow('glovesMask', glovesMask)
    cv2.imwrite('glovesMask.jpg', glovesMask)
    cv2.waitKey(0)    
    cv2.imshow('thresMask', thresMask)
    cv2.imwrite('zeroMask.jpg', zeroMask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()