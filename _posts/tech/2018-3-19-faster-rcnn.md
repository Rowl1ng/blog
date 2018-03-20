---
title:  Faster R-CNN 检测微小物体
layout: blog
background-image: http://static.zybuluo.com/sixijinling/9khganmk3o5gosnvde9l78o2/image_1c8u6de3r12s5oeog9i1ij6d1g9.png

tech: true
date: 2018-3-19 18:26
category: 技术
description:  Faster R-CNN用来识别大件物体效果好，但是小物体效果较差，因此想了一些改进的方法。
tags:
- object detection
---

本文基于[Faster R-CNN的pytorch实现][1]，[原始论文][2]，需要安装：

```
pip install easydict
```

## rpn

### anchor

> a pyramid of regression references。anchor是RPN的windows的大小，在conv features maps的每一个位置使用不同尺度和长宽比的windows提取特征。

在trainval_net.py中，imagenet的`ANCHOR_SCALES`的默认是[4, 8, 16, 32]，`ANCHOR_RATIOS`默认是[0.5,1,2]，但是并不是越多越好，要控制在一定的数量，理由是：

1. anchors多了就会造成faster RCNN的时间复杂度提高，anchors多的最极端情况就是overfeats中sliding windows。
2. 使用多尺度的anchors未必全部对scale-invariant property都有贡献

所以，在使用自己的数据的时候，统计一下gorund truth框的大小，确定大小范围还是很有必要的。

* 重新生成了calcmuba的标注：之前漏掉了单点的，加上了之后fg/bg中fg的比例上升了

> 调max_box_num




loss_rpn_cls是什么？发现降采样改变以后，它奇高

### nms

[讲nms（non-maximum suppression）的文章][3]

## 数据输入：使用自己的数据设计yourdata.py

参考dataset目录下的其他数据集的类进行修改：

- `_classes`：所有框的类别，比较特殊的就是`__background__`

```
# Pascal_VOC
self._classes = ('__background__',  # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')
```
- 关键要修改的函数就是`_load_XXX_annotation(self, index)`（XXX是你的数据集的名字），要实现的功能就是给定image的index，返回所有的bounding box标注。
    - 返回的东西长这样：
```
    return {'width': width,
            'height': height,
            'boxes': np.array(boxes),
            'gt_classes': np.array(gt_classes),
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': np.array(seg_areas)}
```
- `flipped`：默认是horizontal flip，我自己加了vertical flip。
    - 源代码将image_id * 2作为翻转后的图像的index，因此还是要把数据集的index单独存数字id，通过id索引path
    - 同时修改`dataset/minibatch.py`来支持上下翻转
- `gt_overlaps`:Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This overlap value means that crowd "instances" are excluded from training.
- `seg_areas`:

## 大小限制

* need_crop: 窗口大小、rescale到显存支持的一定大小。最长边限制在2100或者短边是1700，导致一块gpu只能跑一个sample
*  [ ] roibatchloader的问题：trim

## 针对小物体

一个完整的VGG16网络长这样：

![VGG16][14]

其中红色部分就是下采样的时刻。原始论文里使用VGG16，因为提feature map只用了最后一次max pooling前面的部分，所以留下来的四次pooling总共下采样是16倍。得到的feature map长这样：

![under sampling][15]

最终的feature映射回原图的话大概长这样：

![此处输入图片的描述][16]

每个小红点之间就是16像素的间隔了，如果要检测特别细小的物体，这么大的下采样九很危险了。于是，为了尽量不破坏细小物体的清晰度，参考[github上关于检测微小物体的讨论][4]，我尝试了两种方案：

### 1. 降低下采样


* 一旦改变下采样，同时还要改变feature_stride和spatial_scale。否则预测框框就变成了边缘的线条。比如VGG16（降采样8倍）：
    * __C.feat_stride': 8”
    * self.spatial_scale to (1/8)
*  vgg16尝试4倍下采样：去掉了stage5的卷积，保障输入图像最短边在1200

#### 剪裁网络

[把卷积层变成全连接层][5]

```
list(model.modules()) # to inspect the modules of your model
my_model = nn.Sequential(*list(model.modules())[:-1]) # strips off last linear layer
```

### 2. 切割原图

- 训练数据切patch：将原图切分为四等份再训练，保障大部分图的清晰度不被压缩。这样一来，标注也要做调整：
    - 一种方式是重新生成新的子图标注，新写一个yourdata.py；
    - 另一种偷懒方式则是修改 yourdata.py的`_load_XXX_anotation(self, index)`，使得读入每个子图，返回的也是每个子图的所有标注框。
- 测试数据为原图，沿用原来的数据读入方式即可。

## models

### VGG16

### Resnet
* resnet的downsample：第一个卷积层就降采样了2倍
* * Resnet18，降低下采样，增加卷积层
![image_1c8u6de3r12s5oeog9i1ij6d1g9.png-183.1kB][6]
检查网络输出

### FPN
![image_1c8u6egich3e1ha41d3o1m22s3g16.png-228.8kB][7]
![image_1c8u6f7esu9i1sl510hs1pnm1qvl1j.png-30.8kB][8]
[原始论文][9]
[FPN的Pytorch实现][10]

### Focal loss

看ICCV那篇focal loss的论文

[Pytorch实现参考][11]

## 其他

negative sampling
* [ ] 作者加lib的方式是_init_paths.py
- caffemodel转pytorch：https://github.com/marvis/pytorch-caffe


## Todo List

- [ ] 可能是resnet的bn_train设置为False，回头改一下
- 加入valid过程，确认是否有过拟合:爆显存了
- 遇到的bug
```
random.choice()
```
## Reference

- 如果想了解object detection的发展史，可以看[Object Detection][12]
- 推荐阅读[Faster R-CNN: Down the rabbit hole of modern object detection][13]


  [1]: https://github.com/jwyang/faster-rcnn.pytorch
  [2]: https://arxiv.org/pdf/1506.01497.pdf
  [3]: https://zhuanlan.zhihu.com/p/31427728
  [4]: https://github.com/rbgirshick/py-faster-rcnn/issues/86
  [5]: https://stackoverflow.com/questions/44146655/how-to-convert-pretrained-fc-layers-to-conv-layers-in-pytorch
  [6]: http://static.zybuluo.com/sixijinling/9khganmk3o5gosnvde9l78o2/image_1c8u6de3r12s5oeog9i1ij6d1g9.png
  [7]: http://static.zybuluo.com/sixijinling/cv2l758k7dyj022k0iinwsm1/image_1c8u6egich3e1ha41d3o1m22s3g16.png
  [8]: http://static.zybuluo.com/sixijinling/elabj076z56xa1xsfbwqyy50/image_1c8u6f7esu9i1sl510hs1pnm1qvl1j.png
  [9]: https://arxiv.org/pdf/1612.03144.pdf
  [10]: https://github.com/jwyang/fpn.pytorch
  [11]: https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
  [12]: https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/
  [13]: https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
  [14]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/vgg.b6e48b99.png
  [15]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/image-to-feature-map.89f5aecb.png
  [16]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/anchors-centers.141181d6.png