import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime
#pragma loop(no_vector)

# loading image
img0 = cv2.imread('D:\Project_lab\zoo.jpg')
# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# 開始測量
start = datetime.datetime.now()
# remove noise
img = cv2.GaussianBlur(gray,(15,15),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=15)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=15)  # y

# 結束測量
end = datetime.datetime.now()

# 輸出結果
print("執行時間：", end - start)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()