# 图像均衡化

![](http://static.zybuluo.com/sixijinling/b5ze3we16g5z2t61qf6k5uwd/Image.png)

[wiki上的说明](https://en.wikipedia.org/wiki/Histogram_equalization)

OpenCV的[Histograms]

```
equ = cv2.equalizeHist(image)
res = np.hstack((image,equ))#两个图片并列拼接在一起，方便和原图对比
plt.imshow(res,cmap='bone')
```

```
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(64,64))
cl1 = clahe.apply(image)
res = np.hstack((image,cl1))
fig = plt.figure(figsize=(10,20))
plt.imshow(res,cmap='bone')
```

```
img2 = cdf[img]
res = np.hstack((img,equ)) #stacking images side-by-side
```
