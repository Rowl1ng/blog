## OpenCV

使用conda安装

```
conda install opencv
```
或者使用pip安装
```
pip install opencv-python
```

## 读入图片

读入灰度图：

```python
img = cv2.imread('13.jpg',0)
```

判断图片是否损坏：

```python
for i, image_path in enumerate(train_image_list):
    test_image = cv2.imread(image_path.encode('utf-8'))
    try:
        test_image.shape
#         print("checked for shape {}".format(test_image.shape))
    except AttributeError:
        print i
        print image_path
        print("shape not found")
```

或者

```
im = cv2.imread(roidb[i]['image'])
assert im is not None, \
    'Failed to read image \'{}\''.format(roidb[i]['image'])
```

### 画图

```
cv2.circle(image_,(1251, 2661),5,(255,255,0),2)
cv2.imwrite('test_1.png', image_)
```

### Resize

注意size的顺序是反的

```
cv2.resize(image, dsize=(width, height), 
                                 interpolation=cv2.INTER_CUBIC)
```

### 矩：Moments

图像矩可以计算图像的质心，面积等等。

```
# 根据图像的矩计算重心 
def find_center(contour):
    M = cv2.moments(contour)
    rx,ry,rw,rh = cv2.boundingRect(contour)
    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print "ROI centroid=", (cx, cy);
    except ZeroDivisionError:
        cx = rx + int(rw / 2)
        cy = ry + int(rh / 2)
        print "ROI centroid=Unknown, use b-box center=", (cx, cy)
    return cx, cy
```

### 获得bbox

注意：输入是int型哦~

```
label_array = np.array(points, dtype=np.int32) #（n, 2）
x1, y1, w, h = cv2.boundingRect(label_array[:, np.newaxis, :]) # (n, 1, 2)
```

### 边缘检测：FindContours

[代码示例](https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/ch21-%E8%BD%AE%E5%BB%93Contours/21-findContour.py)

```python
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    contours,_ = cv2.findContours(roi_mask_8u.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
else:
    _, contours,_ = cv2.findContours(roi_mask_8u.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
```

计算面积

```
area = cv2.contourArea(cnt)
```

### 斑点检测：BlobDetector

[代码示例](https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/ch25-%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B/%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8BSimpleBlobDetector.py)

```
detector = cv2.SimpleBlobDetector()
```



