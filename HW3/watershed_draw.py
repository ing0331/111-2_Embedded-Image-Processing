import numpy as np
import cv2 as cv

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None
        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)           
                self.dirty = True
                self.prev_pt = pt
                self.show()


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
    return img_gray, height, width, img_lbp

class App:
    def __init__(self, fn):
        self.img = cv.imread(fn)
        if self.img is None:
            raise Exception('Failed to load image file: %s' % fn)

        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        print(self.markers)
        self.markers_vis = self.img.copy()

        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255 # type: ignore

        self.auto_update = True
        self.sketch = Sketcher('img', [ self.markers_vis, self.markers], self.get_colors)
               
    def get_colors(self):
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker # type: ignore

    def watershed(self):
        m = self.markers.copy()
        cv.watershed(self.img, m)
        overlay = self.colors[np.maximum(m, 0)] # type: ignore
        oim = cv.imread('mountian_road.jpg')
        vis = cv.addWeighted(oim, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
        cv.imshow('watershed', vis)
 
    def run(self):
        while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
            ch = cv.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
        print(self.markers[:])



        cv.destroyAllWindows()

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'mountian_road.jpg'
    print(__doc__)
    App(fn).run()