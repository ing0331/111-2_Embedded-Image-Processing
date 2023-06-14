import numpy as np
import cv2 as cv
import random as rng

def empty(v):
    pass
cv.namedWindow('trackbar')
cv.resizeWindow('trackbar', 640, 320)
cv.createTrackbar('open inerate', 'trackbar', 1, 15, empty)
cv.createTrackbar('dilate inerate', 'trackbar', 1, 15, empty)
cv.createTrackbar('threshold coef', 'trackbar', 1, 10, empty)

img = cv.imread("D:\\Project_lab\\img_python\\tennis\\tennis_court.jpg")
height, width, channel = img.shape
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

while True:
    open_inerate = cv.getTrackbarPos('open inerate', 'trackbar')
    dilate_inerate  = cv.getTrackbarPos('dilate inerate', 'trackbar')
    coef  = cv.getTrackbarPos('threshold coef', 'trackbar')
    print(open_inerate, dilate_inerate, coef)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    print('thresh', thresh)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = open_inerate)
    print('opening', opening)

   # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations = dilate_inerate)
    print('sure_bg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    print('dist_transform is ', dist_transform)

    ret, sure_fg = cv.threshold(dist_transform, coef/10*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    print('sure_fg', sure_fg)

    unknown = cv.subtract(sure_bg,sure_fg)
    print('unknown', unknown)

    # # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    print(type(markers))
    print('markers', markers)
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 255]

    dist_8u = sure_bg.astype('uint8')
    # # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
    # # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]
    # Visualize the final image
    dst = cv.addWeighted(dst, 0.4, img, 0.6, 0.0)
    cv.imshow('Final Result', img)
    cv.waitKey(0)
img = cv.imread('bad.jpg')
