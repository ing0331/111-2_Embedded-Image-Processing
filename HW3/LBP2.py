import cv2
import numpy as np
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

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray')) # type: ignore
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
    plt.show()
    
def LbpImg(img_bgr):
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    output_list = []
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })    
    show_output(output_list)
    return img_gray, img_lbp

# 讀取圖檔
img = cv2.imread('mountian_road.jpg')
gray, img_lbp = LbpImg(img)   #full screen
height, width, channel = img.shape
print(height, width, channel)

# 建立圖形遮罩
mask1 = np.zeros(gray.shape, np.uint8)
mask1[height-50:height, 110:230] = 255
mask2 = np.zeros(gray.shape, np.uint8)
mask2[height-50:height, 200:320] = 255
mask3 = np.zeros(gray.shape, np.uint8)
mask3[height-50:height, 130:250] = 255
# 計算套用遮罩後的圖形
masked_gray1 = cv2.bitwise_and(gray, gray, mask = mask1)
masked_gray2 = cv2.bitwise_and(gray, gray, mask = mask2)
masked_gray3 = cv2.bitwise_and(gray, gray, mask = mask3)
# 以套用遮罩後的圖計算直方圖
hist_mask1 = cv2.calcHist([img_lbp], [0], mask1, [256], [0, 256])
hist_mask2 = cv2.calcHist([img_lbp], [0], mask2, [256], [0, 256])
hist_mask3 = cv2.calcHist([img_lbp], [0], mask3, [256], [0, 256])

# 繪製結果
plt.subplot(231), plt.imshow(masked_gray1, 'gray')
plt.subplot(232), plt.imshow(masked_gray2, 'gray')
plt.subplot(233), plt.imshow(masked_gray3, 'gray')
plt.subplot(234), plt.plot(hist_mask1, color = 'r') # plt.plot(hist_full),
plt.subplot(235), plt.plot(hist_mask2, color = 'g') # plt.plot(hist_full),
plt.subplot(236), plt.plot(hist_mask3, color = 'r') # plt.plot(hist_full),

plt.xlim([0,256])
plt.show()

cv2.normalize(hist_mask1, hist_mask1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist_mask2, hist_mask2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist_mask3, hist_mask3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
for compare_method in range(4):
    mask1_mask2 = cv2.compareHist(hist_mask1, hist_mask2, compare_method)
    mask2_mask3 = cv2.compareHist(hist_mask2, hist_mask3, compare_method)
    mask1_mask3 = cv2.compareHist(hist_mask1, hist_mask3, compare_method)

    print('Method:', compare_method, 'mask1_mask2, mask2_mask3, mask1_mask3:',\
          mask1_mask2, '/', mask2_mask3, '/', mask1_mask3)
    
# 建立樣本圖形遮罩
# mask_h = np.zeros(gray.shape, np.uint8)
# mask_h[height-17-20:height-20, 230-17:230] = 255
# # 計算套用遮罩後的圖形
# masked_gray1 = cv2.bitwise_and(gray, gray, mask = mask_h)
# # 以套用遮罩後的圖計算直方圖
# hist_mask_h = cv2.calcHist([img_lbp], [0], mask_h, [256], [0, 256])
mask_sel= np.zeros(gray.shape, np.uint8)

mask_h2 = np.zeros(gray.shape, np.uint8)
mask_h2[height-17-50:height-50, 280-17:280] = 255
masked_gray2 = cv2.bitwise_and(gray, gray, mask = mask_h2)
hist_mask_h2 = cv2.calcHist([img_lbp], [0], mask_h2, [256], [0, 256])

x, y = 0, 0
for i in range(21):
    x = 0
    for j in range(31):
        mask_sel[y:y+17, x:x+17] = 255
        hist_mask_sel = cv2.calcHist([img_lbp], [0], mask_sel, [256], [0, 256])

        # maskh_masksel = cv2.compareHist(hist_mask_h, hist_mask_sel, 3)    
        maskh2_masksel= cv2.compareHist(hist_mask_h2, hist_mask_sel, 3)

        if maskh2_masksel > 0.46544704188134297:     #test
            # cv2.rectangle(img, (x, y), (x+17, y+17), (0, 255, 0), -1)
            img[y: y+17, x:x+17]  = [0, 255, 255]
        elif maskh2_masksel >= 0.46544704188134297:
            img[y: y+17, x:x+17] = [0, 255, 255]     
        print('i,j', i, j, end = '')
        print(maskh2_masksel)
        x += 17        
    print('\n')
    y += 17  
cv2.rectangle(img, (230-17,height-17-20), (230,height-20), (255, 0, 0), 1)
cv2.rectangle(img, (280-17, height-17-50),(280 ,height-50 ), (255, 0, 0), 1)

print(type(img))
cv2.imshow('draw', img)
cv2.waitKey(0)
    