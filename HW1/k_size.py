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
img = cv2.GaussianBlur(gray,(3,3),0)
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 15)
dilation = cv2.dilate(img,kernel,iterations = 15)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 結束測量
end = datetime.datetime.now()
# 輸出結果
print("執行時間：", end - start)

plt.subplot(2,2,3),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(erosion,cmap = 'gray')
plt.title('erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(dilation,cmap = 'gray')
plt.title('dilation'), plt.xticks([]), plt.yticks([])

plt.show()
