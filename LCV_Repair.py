# v0001    Honda   20200922    Add DetectVerticalGL, DetectLCD1M2, DetectLCD2M2 function
import numpy as np
import cv2
from os import getcwd, listdir, mkdir
from os.path import join, isdir, isfile, splitext
import InferenceUntils
import math


class CLCVRepair():            
    def __init__(self):
        self.flagDebug = True       
        self.errorLogList=[]
        
    def DetectVerticalGL(self, image, idealMap, boxX, boxY, boxW, boxH):         
        
        '''  
        EXAMPLE:        
            result = CLCVRepair().DetectVerticalGL(image, idealMapCrop, 656, 726, 57, 305)
        
        input:
            image: input image
            idealMap: input cropped ideal map
            boxX: yolo or label box's center x (image coordinate)
            boxY: yolo or label box's center y (image coordinate)
            boxW: yolo or label box's width    (image coordinate)
            boxH: yolo or label box's height   (image coordinate)   

        output:
            result: repair line (['GLOpen', x1, y1, x2, y2])
            self.errorLogList: error message            
        '''
        
        #---Local variables---#
        result = []
        self.errorLogList = []       
        TFTXPositionList = []
        TFTYPositionList = []
        TFTHeightList = []
        x1, y1, x2, y2 = -1, -1, -1, -1        
        InferenceUntils.debug('CLCVRepair initialize success', self.flagDebug)        
        
        #---Filter TFT area---#
        bgr = [150, 200, 0]
        thresh = 10        
        minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
        maskBGR = cv2.inRange(idealMap,minBGR,maxBGR)
        maskBGR = cv2.erode(maskBGR, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))) 
        InferenceUntils.debug('Filter TFT success', self.flagDebug)
        
        #---Find TFT contours---#
        contours,hierarchy = cv2.findContours(maskBGR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)            
        for c in contours:     
            x,y,w,h = cv2.boundingRect(c)
            TFTXPositionList.append(int(x + w / 2))
            TFTYPositionList.append(int(y + h / 2))
            TFTHeightList.append(h)
        if len(TFTHeightList) == 0:
            InferenceUntils.debug('Find TFT contours fail', self.flagDebug)
        else:
            InferenceUntils.debug('Find TFT contours success', self.flagDebug)
        
        #---Find X Position---#            
        TFTXPositionList = np.asarray(TFTXPositionList)   
        TFTYPositionList = np.asarray(TFTYPositionList)    
        TFTHeightList = np.asarray(TFTHeightList)   
        try:
            GLXPosition = int(TFTXPositionList[np.argmin(abs(TFTXPositionList - boxX))])     
            x1 = GLXPosition
            x2 = x1
            InferenceUntils.debug('Find X Position success', self.flagDebug)
        except:        
            InferenceUntils.debug('Find X Position fail\n', self.flagDebug)
            return ['GLOpen', x1, y1, x2, y2]            
        
        #---Find Y position---#            
        y1 = int(boxY - boxH / 2 - 15)
        y2 = int(boxY + boxH / 2 + 15)    
        TFTHeight = np.median(TFTHeightList) 
        
        GLYPosition = int(TFTYPositionList[np.argmin(abs(TFTYPositionList - y1))])
        if y1 - GLYPosition > - TFTHeight * 2 and y1 - GLYPosition < TFTHeight:
            y1 = int(GLYPosition - TFTHeight * 2)
            
        GLYPosition = int(TFTYPositionList[np.argmin(abs(TFTYPositionList - y2))])  
        if y2 - GLYPosition > - TFTHeight * 2 and y2 - GLYPosition < TFTHeight:
            y2 = int(GLYPosition + TFTHeight) 
            
        y1 = 0  if y1 < 0 else y1
        y2 = image.shape[0] - 1  if y2 > image.shape[0] - 1 else y2 
        
        result = ['GLOpen', x1, y1, x2, y2]
        InferenceUntils.debug('Find Y Position success', self.flagDebug)
        InferenceUntils.debug('Result is (' + str(x1) + ', ' + str(y1) + ') (' + str(x2) + ', ' + str(y2) + ')\n', self.flagDebug)
         
        return result  
    
    
    def DetectLCD1M2(self, image, boxX, boxY, boxW, boxH):    

        '''  
        EXAMPLE:        
            result = CLCVRepair().DetectLCD1M2(image, 656, 726, 57, 305)
        
        input:
            image: input image            
            boxX: yolo or label box's center x (image coordinate)
            boxY: yolo or label box's center y (image coordinate)
            boxW: yolo or label box's width    (image coordinate)
            boxH: yolo or label box's height   (image coordinate)   

        output:
            result: repair line (['LCV-No-Gate', x1, y1, x2, y2])
            self.errorLogList: error message

        except:
            result: repair line (['LCV-No-Gate', -1, -1, -1, -1])
        '''
        
        #---Local variables---#
        result = []
        extend = 20
        roiW = (boxW + extend) * 1.5   
        roiH = (boxH + extend) * 1.5        
        
        #---Adjust brightness---# 
        brightness = np.sum(image) / (255 * image.shape[0] * image.shape[1])         
        ratio = brightness / 1.3
        image = cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0) 
        
        #---Get canny edge---# 
        b,g,r = cv2.split(image)
        imageDiff = cv2.subtract(g, r)
        _, imageThr1 = cv2.threshold(b,120,255,cv2.THRESH_BINARY)        
        _, imageThr2 = cv2.threshold(g,120,255,cv2.THRESH_BINARY)
        _, imageThr3 = cv2.threshold(r,120,255,cv2.THRESH_BINARY)        
        _, imageThr4 = cv2.threshold(imageDiff,50,255,cv2.THRESH_BINARY_INV)
        _, imageThr5 = cv2.threshold(imageDiff,5,255,cv2.THRESH_BINARY)
        imageThr = cv2.bitwise_and(imageThr1, imageThr2)
        imageThr = cv2.bitwise_and(imageThr, imageThr3)
        imageThr = cv2.bitwise_and(imageThr, imageThr4)
        imageThr = cv2.bitwise_and(imageThr, imageThr5)  
        imageCanny = cv2.Canny(imageThr, 100, 200, apertureSize=3)   
        
        #---Find lines in ROI---#
        x1 = int(boxX - roiW / 2) if boxX - roiW / 2 > 0 else 0 
        x2 = int(boxX + roiW / 2) if boxX + roiW / 2 < image.shape[1] else image.shape[1] - 1 
        y1 = int(boxY - roiH / 2) if boxY - roiH / 2 > 0 else 0 
        y2 = int(boxY + roiH / 2) if boxY + roiH / 2 < image.shape[0] else image.shape[0] - 1
        imageThrROI = imageThr[y1:y2,x1:x2] 
        imageCannyROI = imageCanny[y1:y2, x1:x2]     
        lines = cv2.HoughLinesP(imageCannyROI, 1, np.pi/180, 15, minLineLength=5, maxLineGap=max(roiW, roiH))
        
        #---Calculate the longest line's slope---# 
        lMax = 0
        m = 0        
        try:            
            for x in range(0, len(lines)):
                for a1,b1,a2,b2 in lines[x]:                    
                    if ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5 > lMax:
                        lMax = ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5 
                        if a2 - a1 != 0:
                            m = (b2 - b1) / (a2 - a1)                    
                        else:    # Vertical line
                            m = -99                        
        except:
            pass   
        
        #---Draw the line by point slope form y - y_avg = m * (x - x_avg)---# 
        if lMax != 0:   
            if abs(m) > 1:    # Vertical line
                listX = []    # Find the new center of x coordinate 
                for i in range(0, imageThrROI.shape[0]):      
                    m2List = imageThrROI[i,:]                    
                    if any(m2List):                    
                        xMean = np.mean(np.where(m2List == 255))  
                        listX.append(int(xMean + 0.5))
                
                if any(listX): 
                    boxX = int(sum(listX) / len(listX) + boxX - roiW / 2 + 0.5)
                                               
                y1 = int(boxY - (boxH + extend) / 2)    if boxY - (boxH + extend) / 2 > 0 else 0
                x1 = int((y1 - boxY) / m + boxX)        if (y1 - boxY) / m + boxX > 0 else 0
                y2 = int(boxY + (boxH + extend) / 2)    if boxY + (boxH + extend) / 2 < image.shape[0] else image.shape[0] - 1                 
                x2 = int((y2 - boxY) / m + boxX)        if (y2 - boxY) / m + boxX < image.shape[1] else image.shape[1] - 1
                
            elif abs(m) < 1:    # Horizontal line             
                listY = []    # Find the new center of y coordinate        
                for i in range(0, imageThrROI.shape[1]):      
                    m2List = imageThrROI[:,i]                    
                    if any(m2List):                        
                        yMean = np.mean(np.where(m2List == 255))
                        listY.append(int(yMean + 0.5))
                
                if any(listY):                 
                    boxY = int(sum(listY) / len(listY) + boxY - roiH / 2 + 0.5)                 
                
                x1 = int(boxX - (boxW + extend) / 2)    if boxX - (boxW + extend) / 2 > 0 else 0
                y1 = int((x1 - boxX) * m + boxY)        if (x1 - boxX) * m + boxY > 0 else 0
                x2 = int(boxX + (boxW + extend) / 2)    if boxX + (boxW + extend) / 2 < image.shape[1] else image.shape[1] - 1
                y2 = int((x2 - boxX) * m + boxY)        if (x2 - boxX) * m + boxY < image.shape[0] else image.shape[0] - 1          
                         
            elif abs(m) == 1:    # 45 degree line
                x1 = int(boxX - (boxW + extend) / 2)        if boxX - (boxW + extend) / 2 > 0 else 0
                y1 = int(boxY - m * (boxH + extend) / 2)    if boxY - m * (boxH + extend) / 2 > 0 else 0
                x2 = int(boxX + (boxW + extend) / 2)        if boxX + (boxW + extend) / 2 < image.shape[1] else image.shape[1] - 1
                y2 = int(boxY + m * (boxH + extend) / 2)    if boxY + m * (boxH + extend) / 2 < image.shape[0] else image.shape[0] - 1
        else:
            x1 = -1
            y1 = -1
            x2 = -1
            y2 = -1
            
        result = ['LCV-No-Gate', x1, y1, x2, y2]
        
        return result  
    
    def DetectLCD2M2(self, image, boxX, boxY, boxW, boxH):
        
        '''  
        EXAMPLE:        
            result = CLCVRepair().DetectLCD2M2(image, 656, 726, 57, 305)
        
        input:
            image: input image            
            boxX: yolo or label box's center x (image coordinate)
            boxY: yolo or label box's center y (image coordinate)
            boxW: yolo or label box's width    (image coordinate)
            boxH: yolo or label box's height   (image coordinate)   

        output:
            result: repair line (['M2-Open', x1, y1, x2, y2])
            self.errorLogList: error message

        except:
            result: repair line (['M2-Open', -1, -1, -1, -1])
        '''
        
        #---Local variables---#
        result = []
        extend = 20
        roiW = (boxW + extend) * 2
        roiH = (boxH + extend) * 2  
        
        #---Get canny edge---#       
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        imageGaussian = cv2.GaussianBlur(imageGray,(5,5),0) 
        _, imageThr = cv2.threshold(imageGaussian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)        
        imageErode = cv2.erode(imageThr, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))         
        imageCanny = cv2.Canny(imageErode,150,200,apertureSize = 3)        
        
        #---Find lines in ROI---#
        x1 = int(boxX - roiW / 2) if boxX - roiW / 2 > 0 else 0 
        x2 = int(boxX + roiW / 2) if boxX + roiW / 2 < image.shape[1] else image.shape[1] - 1 
        y1 = int(boxY - roiH / 2) if boxY - roiH / 2 > 0 else 0 
        y2 = int(boxY + roiH / 2) if boxY + roiH / 2 < image.shape[0] else image.shape[0] - 1
        imageThrROI = imageThr[y1:y2,x1:x2] 
        imageCannyROI = imageCanny[y1:y2, x1:x2]     
        lines = cv2.HoughLinesP(imageCannyROI, 1, np.pi/180, 15, minLineLength=5, maxLineGap=max(roiW, roiH))
        
        #---Calculate the longest line's slope---# 
        lMax = 0
        m = 0        
        try:            
            for x in range(0, len(lines)):
                for a1,b1,a2,b2 in lines[x]:                    
                    if ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5 > lMax:
                        lMax = ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5 
                        if a2 - a1 != 0:
                            m = (b2 - b1) / (a2 - a1)                    
                        else:    # Vertical line
                            m = -99                        
        except:
            pass    
        
        #---draw the line---# 
        if lMax != 0:            
            if abs(m) > 1:    # Vertical line
                listX = []    # Find the new center of x coordinate 
                for i in range(0, imageThrROI.shape[0]):         
                    m2List = imageThrROI[i,:]                   
                    if any(m2List):
                        xMedian = np.median(np.where(m2List == 255))    # Find the medium of each row
                        listX.append(int(xMedian + 0.5))
                   
                if any(listX):                    
                    counts = np.bincount(listX)    # Find the mode 
                    boxX = int(np.argmax(counts) + boxX - roiW / 2 + 0.5) 
                
                x1 = int(boxX)                          if boxX > 0 else 0
                y1 = int(boxY - (boxH + extend) / 2)    if boxY - (boxH + extend) / 2 > 0 else 0  
                x2 = int(boxX)                          if boxX < image.shape[1] else image.shape[1] - 1 
                y2 = int(boxY + (boxH + extend) / 2)    if boxY + (boxH + extend) / 2 < image.shape[0] else image.shape[0] - 1               
                
            elif abs(m) < 1:    # Horizontal line             
                listY = []    # Find the new center of y coordinate        
                for i in range(0, imageThrROI.shape[1]):      
                    m2List = imageThrROI[:,i]                                        
                    if any(m2List): 
                        yMedian = np.median(np.where(m2List == 255))    # Find the medium of each col
                        listY.append(int(yMedian + 0.5))
                
                if any(listY):                   
                    counts = np.bincount(listY)    # Find the mode   
                    boxY = int(np.argmax(counts) + boxY - roiH / 2 + 0.5) 
                
                x1 = int(boxX - (boxW + extend) / 2)    if boxX - (boxW + extend) / 2 > 0 else 0
                y1 = int(boxY)                          if boxY > 0 else 0
                x2 = int(boxX + (boxW + extend) / 2)    if boxX + (boxW + extend) / 2 < image.shape[1] else image.shape[1] - 1 
                y2 = int(boxY)                          if boxY < image.shape[0] else image.shape[0] - 1
        else:
            x1 = -1
            y1 = -1
            x2 = -1
            y2 = -1
            
        result = ['M2-Open', x1, y1, x2, y2]
        
        return result  
        
    def Check(self):
        return 'Load LCVRepair success\n'
    
if __name__ == '__main__':
    
    #---Settings---#
    testMode = 'PEP3-1'    # Select: PEP1, PEP3-1, PEP3-2
    Worker = CLCVRepair()
    Worker.flagDebug = True    
    imageFolder = join(getcwd(), 'test')    # Input test folder path   
    if testMode == 'PEP1':
        idealMap = cv2.imread('L7A_IdealMap.jpg')    # Input non-crop ideal map
    else:
        idealMap = None

    #---Make image file list from folder---#
    includedExtensions = ['jpg','jpeg', 'bmp', 'png']
    imageList = [file for file in listdir(imageFolder)
                   if any(file.endswith(ext) for ext in includedExtensions)]    
    if not isdir(imageFolder + '_result'):
        mkdir(imageFolder + '_result')   
    
    #---Start iteration---#
    countNum = 1
    for imgfile in imageList:            
        imagePath = join(imageFolder, imgfile)
        txtPath = imagePath.replace(splitext(imagePath)[-1], '.txt')         
    
        if isfile(txtPath):
            print('(' + str(countNum) + '/' + str(len(imageList)) + ') ' + imgfile)
            countNum += 1
            image = cv2.imread(imagePath)               
            
            with open(txtPath, 'r', encoding='utf8') as fp: 
                line = fp.readline()
                boxX = int(float(line.split(' ')[1]) * image.shape[1])
                boxY = int(float(line.split(' ')[2]) * image.shape[0])
                boxW = int(float(line.split(' ')[3]) * image.shape[1])
                boxH = int(float(line.split(' ')[4]) * image.shape[0])                
            
            if testMode == 'PEP1':    # Match template and crop ideal map
                imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                idealMapGray = cv2.cvtColor(idealMap, cv2.COLOR_BGR2GRAY) 
                res = cv2.matchTemplate(imageGray,idealMapGray,cv2.TM_CCOEFF_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)            
                topLeft = maxLoc
                bottomRight = (topLeft[0] + image.shape[1], topLeft[1] + image.shape[0])
                idealMapCrop = idealMap[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]              
                result = Worker.DetectVerticalGL(image, idealMapCrop, boxX, boxY, boxW, boxH)    # Example           
            elif testMode == 'PEP3-1':
                result = Worker.DetectLCD1M2(image, boxX, boxY, boxW, boxH)    # Example 
            elif testMode == 'PEP3-2':
                result = Worker.DetectLCD2M2(image, boxX, boxY, boxW, boxH)    # Example     
            
            if result[1] != -1:    # Print repair result
                cv2.line(image,(int(result[1]),int(result[2])),(int(result[3]),int(result[4])),(0,0,255),2)
                cv2.rectangle(image, (int(boxX - boxW / 2), int(boxY - boxH / 2)), (int(boxX + boxW / 2), int(boxY + boxH / 2)) ,(255, 0, 0), 2)  
                cv2.imwrite(join(imageFolder + '_result', imgfile.split('.')[0] + '_result.jpg'), image)
            else:
                cv2.rectangle(image, (int(boxX - boxW / 2), int(boxY - boxH / 2)), (int(boxX + boxW / 2), int(boxY + boxH / 2)) ,(255, 0, 0), 2)  
                cv2.imwrite(join(imageFolder + '_result', imgfile.split('.')[0] + '_result.jpg'), image)
            
        