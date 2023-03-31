import numpy as np
import cv2 as cv
import random as rng
import matplotlib.pyplot as plt

def GetPixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''   
    center = img[x][y]
    val_ar = []
    val_ar.append(GetPixel(img, center, x-1, y+1))     # top_right
    val_ar.append(GetPixel(img, center, x, y+1))       # right
    val_ar.append(GetPixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(GetPixel(img, center, x+1, y))       # bottom
    val_ar.append(GetPixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(GetPixel(img, center, x, y-1))       # left
    val_ar.append(GetPixel(img, center, x-1, y-1))     # top_left
    val_ar.append(GetPixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]   #權重
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]     
    return val    
def LbpImg(img_bgr):
    height, width, channel = img_bgr.shape
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp

def empty(v):
    pass

cv.namedWindow('trackbar')
cv.resizeWindow('trackbar', 640, 320)
cv.createTrackbar('open inerate', 'trackbar', 1, 15, empty)
cv.createTrackbar('dilate inerate', 'trackbar', 1, 15, empty)
cv.createTrackbar('threshold coef', 'trackbar', 1, 10, empty)

img = cv.imread('mountian_road.jpg')
height, width, channel = img.shape
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

img_lbp = LbpImg(img)
mask_h2 = np.zeros(gray.shape, np.uint8)
mask_h2[height-17-50:height-50, 280-17:280] = 255
hist_mask_h2 = cv.calcHist([img_lbp], [0], mask_h2, [256], [0, 256])
mask_sel= np.zeros(gray.shape, np.uint8)

while True:
    open_inerate = cv.getTrackbarPos('open inerate', 'trackbar')
    dilate_inerate  = cv.getTrackbarPos('dilate inerate', 'trackbar')
    coef  = cv.getTrackbarPos('threshold coef', 'trackbar')
    print(open_inerate, dilate_inerate, coef)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    print('thresh', thresh)

    x, y = 0, 0
    for i in range(21):
        x = 0
        for j in range(31):
            mask_sel[y:y+17, x:x+17] = 255
            hist_mask_sel = cv.calcHist([img_lbp], [0], mask_sel, [256], [0, 256])
            maskh2_masksel= cv.compareHist(hist_mask_h2, hist_mask_sel, 3)
            if maskh2_masksel <= 0.46024200644736:     #test
                thresh[y: y+17, x:x+17] = 0
            elif maskh2_masksel >= 0.46544704188134297 or y < int(height/3):
                thresh[y: y+17, x:x+17] = 255        
            x += 17        
        y += 17        

    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = open_inerate)
    print('opening', opening)
        
   # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations = dilate_inerate)
    print('sure_bg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    print('dist_transform is ', dist_transform)
    print('\n')

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
    cv.imshow('Final Result', dst)
    cv.waitKey(0)
    img = cv.imread('D:\Project_lab\mountian_road.jpg')
