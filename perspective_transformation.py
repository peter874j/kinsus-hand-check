import cv2
import numpy as np
import argparse
import yaml
import os

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # or use this function
    # M = cv2.findHomography(pts, dst)[0]

    print("angle of rotation: {}".format(np.arctan2(-M[1, 0], M[0, 0]) * 180 / np.pi))

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# define 4 points for ROI
def selectROI(event, x, y, flags, param):
    global imagetmp, roiPts

    if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append([x, y])
        print(x, y)
        cv2.circle(imagetmp, (x, y), 2, (0, 255, 0), -1)
        if len(roiPts) < 1:
            cv2.line(imagetmp, roiPts[-2], (x, y), (0, 255, 0), 2)
        if len(roiPts) == 4:
            cv2.line(imagetmp, roiPts[0], (x, y), (0, 255, 0), 2)

        cv2.imshow("image", imagetmp)
        print("select ROI")

def load_yaml(ymlfile):
    with open(ymlfile, "r") as ymlfile:
        contentDict = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return contentDict

def dump_yaml(ymlfile, pointsList):
    contentDict = dict()
    contentDict["four_points"] = pointsList
    with open(ymlfile, 'w') as file:
        documents = yaml.dump(contentDict, file)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remark', action='store_true', default=False, help='mark 4 points to perspective transform')
    opt = parser.parse_args()
    return opt

def shi_tomasi(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x,y), 3, (0, 0, 255), -1)

def segmentation_(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite('gray.jpg', gray)
    cv2.imwrite('HSV.jpg', HSV)
    ret, thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  
    cv2.imshow("gray", gray)
    cv2.waitKey()
    cv2.imshow("thresh", thresh)
    cv2.waitKey()
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow("opening", opening)
    cv2.waitKey()
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow("sure_bg", sure_bg)
    cv2.waitKey()
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    cv2.imshow("sure_fg", sure_fg)
    cv2.waitKey()
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow("unknown", unknown)
    cv2.waitKey()
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    cv2.imshow("markers", markers)
    cv2.waitKey()
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    cv2.imshow("img", img)
    cv2.waitKey()

def segmentation(img):
    #convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.equalizehist(img)
    # pixel = hsv[66, 686]
    #obtain the grayscale image of the original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizehist(gray)
    #set the bounds for the red hue
    # lowerPink = np.array([150, 153, 208])
    # upperPink = np.array([123, 135, 171])
    b, g, r = cv2.split(img)
    imageDiff = cv2.subtract(g, b)
    _, imageThr1 = cv2.threshold(imageDiff, 5, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('imageThr1.jpg', imageThr1)
    imageThr = cv2.bitwise_and(imageThr1, gray)
    cv2.imwrite('imageThr.jpg', imageThr)
    lowerPinkRGB = np.array([85, 80, 135])
    upperPinkRGB = np.array([150, 180, 190])    
    lowerPinkHSV = np.array([0, 72, 139])
    upperPinkHSV = np.array([175, 103, 230])
    print(hsv[67, 138])
    print(hsv[496, 758])
    #create a mask using the bounds set
    maskRGB = cv2.inRange(img, lowerPinkRGB, upperPinkRGB)
    maskHSV = cv2.inRange(img, lowerPinkHSV, upperPinkHSV)
    imageThr = cv2.bitwise_and(imageThr1, maskRGB)
    cv2.imwrite('imageThr.jpg', imageThr)
    cv2.imwrite('pinkRGB.jpg', maskRGB)
    cv2.imwrite('pinkHSV.jpg', maskHSV)
    #create an inverse of the mask
    mask_inv = cv2.bitwise_not(maskRGB)
    #Filter only the red colour from the original image using the mask(foreground)
    res = cv2.bitwise_and(img, img, mask=maskRGB)
    #Filter the regions containing colours other than red from the grayscale image(background)
    background = cv2.bitwise_and(gray, gray, mask = mask_inv)
    #convert the one channelled grayscale background to a three channelled image
    background = np.stack((background,)*3, axis=-1)
    #add the foreground and the background
    added_img = cv2.add(res, background)

    #create resizable windows for the images
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("added", cv2.WINDOW_NORMAL)
    cv2.namedWindow("back", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask_inv", cv2.WINDOW_NORMAL)
    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)

    #display the images
    cv2.imshow("back", background)
    cv2.imshow("mask_inv", mask_inv)
    cv2.imshow("added",added_img)
    cv2.imshow("mask", maskRGB)
    cv2.imshow("gray", gray)
    cv2.imshow("hsv", hsv)
    cv2.imshow("res", res)

    if cv2.waitKey(0):
        cv2.destroyAllWindows()

def main():
    ymlfile = r'.\config.yaml'
    global imagetmp, roiPts
    roiPts = []
    opt = parse_argument()
    remarkFlag = opt.remark
    remarkFlag = True
    # image = cv2.imread("hands_standard.jpg")
    imgfolder = r'D:\_Project\E-fence\Camera_Perspective_Transformation\images'
    outfolder = r'D:\_Project\E-fence\Camera_Perspective_Transformation\image_output'
    for root, dirs, files in os.walk(imgfolder):
        for f in files: 
            if f[-2:]=='pg':
                roiPts = []
                image = cv2.imread(os.path.join(root, f))
                ### read 4 points from Mouse clicking
                if remarkFlag:
                    imagetmp = image.copy()
                    cv2.namedWindow("image")
                    cv2.setMouseCallback("image", selectROI)

                    while len(roiPts) < 4:
                        cv2.imshow("image", imagetmp)
                        cv2.waitKey(500)
                    dump_yaml(ymlfile, roiPts)
                ### read 4 points from yaml
                else:
                    pointsDict = load_yaml(ymlfile)
                    roiPts = pointsDict["four_points"]

                roiPts = np.array(roiPts, dtype=np.float32)
                warped = four_point_transform(image, roiPts)
                # cv2.imwrite('perspect_.jpg', warped)
                # segmentation(warped)
                # shi_tomasi(warped)
                # cv2.imshow("Warped", warped)
                # cv2.waitKey()
                f = f.replace('hands', 'perspective_')
                cv2.imwrite(os.path.join(outfolder, f), warped)
                cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()