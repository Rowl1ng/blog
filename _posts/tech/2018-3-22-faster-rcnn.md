---
title:  Detectron代码分析：从Faster R-CNN 到 Mask R-CNN
layout: blog
background-image: http://static.zybuluo.com/sixijinling/9khganmk3o5gosnvde9l78o2/image_1c8u6de3r12s5oeog9i1ij6d1g9.png
tech: true
istop: true
mathjax: true
date: 2018-6-4 20:15
category: tech
description:  基于pytorch版Detectron
tags:
- object detection
version:
  current: 简体中文
versions:
  - 简体中文: #
  - English: #
---

{% include post-version-selector.html %}

本文mask rcnn的部分基于[Detectron.pytorch][1]（参考了[Faster R-CNN的pytorch实现][3]），[Detectron][2]（使用caffe2）是face book开源的的各种目标检测算法的打包（比如mask rcnn、FPN神马的），可以学习一下。

需要安装：

- pytorch > 0.3.0
- pycocotools

```
pip install easydict cython
```

# 代码结构


```
- configs
    - 各种网络的配置文件.yml
- lib
    - core：
        - config.py: 定义了通用型rcnn可能用到的所有超参
        - test_engine.py: 整个测试流程的控制器
        - test.py
    - dataset: 原始数据IO、预处理
        - your_data.py：在这里定义你自己的数据、标注读取方式
        - roidb.py
    - roi_data：数据工厂，根据config的网络配置生成需要的各种roi、anchor等
        - loader.py
        - rpn.py: 生成RPN需要的blob
        - data_utils.py: 生成anchor
    - modeling: 各种网络插件，rpn、mask、fpn等
        - model_builder.py：构造generalized rcnn
        - ResNet.py: Resnet backbone相关
        - FPN.py：RPN with an FPN backbone
        - rpn_heads.py：RPN and Faster R-CNN outputs and losses
        - mask_rcnn_heads：Mask R-CNN outputs and losses
    - utils：小工具
- tools
    - train_net.py: 训练
    - test_net.py: 测试
```

# 数据输入

## 图像预处理


原项目的`lib/datasets`中提供了imagenet、COCO等通用数据集的调用类，如果想使用自己的数据的话就需要仿照着设计yourdata.py。几个要注意的点：

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
    - 返回的roidb长这样：
```
    return {'width': width,
            'height': height,
            'boxes': np.array(boxes),
            'gt_classes': np.array(gt_classes),
            'gt_overlaps': overlaps,
            'flipped': False, # 用于data augmentation
            'seg_areas': np.array(seg_areas)}
   # 得到roidb之后还会计算：
   'max_classes' # 和哪个class重合最大？
   'max_overlaps' # 和该class重合率[0,1]
   # 后面的采样环节会用到
```
- `gt_overlaps`：有些数据集（比如COCO）中一个bbox囊括了好几个对象，称之为`crowd box`，训练时需要把他们移除，移除的手段就是把它们的overlap (with all categories)设为负值（比如-1）。
- `seg_areas`: mask rcnn根据segment的区域大小排序

说一下[dataset/roidb.py][4]这个文件，里面最重要的就是`combined_roidb_for_training`，它是训练数据的“组装车间”，当需要同时训练多个数据集时尤为方便。通过调用每个数据集的get_roidb()方法获得各个数据集的roidb，并对相应数据集进行augmentation，这里主要是横向翻转（flipped），最后返回的是augmentation后打包在一起的roidb（前半部分是原始图像+标注，后半部分是原始图像+翻转后的标注）。

- 减去的均值是固定的$$3 \times 1$$向量（对train和test都是一样）；
- Agmentation: 默认是Horizontal Flip（可以仿照加Vertically Flip）
- 控制训练时每块GPU上每个minibatch的ratio都相同:TRAIN.`ASPECT_GROUPING` = True
- 计算bbox regrssion $$\delta$$。需要的关键文件：[utils/boxes.py][5]
    - bbox_transform_inv：通过proposal box和groundtruth box 计算bbox regrssion $$\delta$$
    - MODEL.`BBOX_REG_WEIGHTS` :加权用，默认是(10., 10., 5., 5.)

[roi_data/loader.py][6]中的RoiDataLoader对上面处理完的roidb“加工”成 data，主要通过get_minibatch获得某张图的一个minibatch。可以看minibatch.py的实现。首先会初始化所需blob的name list，比如FPN对应的list如下：
```
['roidb',
'data', # （1，3，2464，2016）
'im_info',

'rpn_labels_int32_wide_fpn2', # （1，3，752，752）
'rpn_labels_int32_wide_fpn3', # （1，3，376，376）
'rpn_labels_int32_wide_fpn4',  # （1，3，188，188）
'rpn_labels_int32_wide_fpn5', # （1，3，94，94）
'rpn_labels_int32_wide_fpn6', # （1，3，47，47）

'rpn_bbox_targets_wide_fpn2', # （1，12，752，752）
'rpn_bbox_targets_wide_fpn3',
'rpn_bbox_targets_wide_fpn6',
'rpn_bbox_targets_wide_fpn4',
'rpn_bbox_targets_wide_fpn5',

'rpn_bbox_outside_weights_wide_fpn2', # （1，12，752，752）
'rpn_bbox_outside_weights_wide_fpn3',
'rpn_bbox_outside_weights_wide_fpn4',
'rpn_bbox_outside_weights_wide_fpn5',
'rpn_bbox_outside_weights_wide_fpn6',

'rpn_bbox_inside_weights_wide_fpn6',
'rpn_bbox_inside_weights_wide_fpn5',
'rpn_bbox_inside_weights_wide_fpn4',
'rpn_bbox_inside_weights_wide_fpn3',
'rpn_bbox_inside_weights_wide_fpn2'] # （1，12，752，752）
```

在获得list后，使用[roi_data/rpn.py][7]的add_rpn_blobs来填上对应的blob。`_get_rpn_blobs`的流程：

- 生成anchor
- TRAIN.`RPN_STRADDLE_THRESH`：筛除超出image范围的RPN anchor，默认是0.
- 计算anchor label： positive：label=1； negative：label=0；don't care: label=-1
    - 计算anchor和gt box的overlap
        - Fg label（positive）：和每个gt重合率最大的那个anchor；超过TRAIN.`RPN_POSITIVE_OVERLAP`的anchor
    - 控制数量，采样positive 和 nagative label，超过的将label设为-1
        - fg/正样本：
            - 总数 = `TRAIN.FG_FRACTION` * `TRAIN.BATCH_SIZE_PER_IM`
            - 条件：> `TRAIN.FG_THRESH`，随机选择达到条件的fg，数量<=总数：fg_inds = np.where(`max_overlaps` >= cfg.`TRAIN.FG_THRESH`)[0]
        - bg/负样本：
            - 总数 = `TRAIN.BATCH_SIZE_PER_IM` - 正样本总数
            - 条件:[`BG_THRESH_LO`, `BG_THRESH_HI`)
- bbox regression loss:loss(x) = weight_outside * L(weight_inside * x)
    - bbox_inside_weights: bbox regression只用positive example来训，所以只需把positive的weight设为1.0，其他设为0即可（只有那些分类正确了的box才能参与，分类错误的直接不考虑了）
    - bbox_outside_weights: bbox regression loss是对minibatch中的图片数取平均，

![此处输入图片的描述][8]

这里是针对各个类别的。也就是说，在后面的classification layer中，这个bbox_inside_weights起一个mask的作用，这样就只计算fg的，不管bg的，但是算分类loss的时候还是都考虑。

> Bbox regression loss has the form:
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise

> 代码实现上：使用了mask array将for循环操作转化成了矩阵乘法。mask array标注了每个anchor的正确物体类别。




## 正负采样


关键文件：[roi_data/minibatch.py][9]

- 考虑到一块gpu的显存有限，我们往往需要将图像rescale的一定大小。 `targetSize` 和 `maxSize` 默认是 600 和 1000 ，根据显存大小和图片大小来设计就好（比如最长边限制在2100或者短边是1700的时候一块gpu只能跑一个sample）。
    - `TRAIN.SCALES`：一个list，默认是[600,]，如果有多个值，在采样的时候是随机分配给image的：`np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)`

关键文件： [roi_data/fast_rcnn.py][10]

- _sample_rois：随机采样fg/bg examples，控制数量的方法和之前一样

Rescale的基本逻辑如下图：

![此处输入图片的描述][11]

这一步在**决定使用的anchor size**时一定要考虑进去，github上有人写过基于自己数据的分析脚本，基本思路是还原rescale的过程，分析rescale factor，估计一下roi的大小，从而决定anchor size。

计算所有ROI和所有ground truth的max overlap，从而判断是fg还是bg。这里用到了两个参数：

- `TRAIN.FG_THRESH`：用来判断fg ROI（default：0.5）
- `TRAIN.BG_THRESH_LO` ~ `TRAIN.BG_THRESH_HI`：这个区间的是bg ROI。(default 0.1, 0.5 respectively)

这样的设计可以看作是 “hard negative mining” ，用来给classifier投喂更难的bg样本。



输入:

- ROIs produced by the proposal layer
- ground truth information

输出:

- Selected foreground and background ROIs that meet overlap criteria.
- Class specific target regression coefficients for the ROIs

Parameters:

- `TRAIN.BATCH_SIZE`: (default 128) 所选fg和bg box的最大数量.
- `TRAIN.FG_FRACTION`: (default 0.25). fg box 不能超过 BATCH_SIZE*FG_FRACTION

# Backbone

在初始化generalized rcnn时，首先要选择backbone（也许可以意译为“骨干网”）。通过cfg.MODEL.`CONV_BODY`即可选择backbone：

```
self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)() # 如果是FPN，会在这一步直接基于backbone完成FPN的构造
```

## Resnet

Resnet.py打包了各种resnet的backbone，比如ResNet50_conv5_body：

```
def ResNet50_conv5_body():
    return ResNet_convX_body((3, 4, 6, 3)) # block_counts: 分别对应res2、res3、res4、res5
```
![此处输入图片的描述][12]

Resnet中的Bottleneck：

![此处输入图片的描述][13]

## FPN

[原始论文][14]

可以选择是否用于RoI transform，是否用于RPN：

1. **feature extraction**：fpn.py中的fpn类将backbone“加工”为FPN。比如当config里的CONV_BODY: FPN.fpn_ResNet50_conv5_body，实际要做的就是先初始化Resnet.ResNet50_conv5_body，再交给fpn得到fpn_ResNet50_conv5_body。
2. **RPN with FPN backbone**: 如果rpn使用FPN，那么FPN的每个level都会做RPN，生成相应大小的anchor，返回相应的cls_score和bbox_pred。rpn_heads,FPN.fpn_rpn_outputs

![image_1c8u6egich3e1ha41d3o1m22s3g16.png-228.8kB][15]

![image_1c8u6f7esu9i1sl510hs1pnm1qvl1j.png-30.8kB][16]

还提供了一个选择：OnlyP6

```
# FPN is enabled if True
__C.FPN.FPN_ON = False

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# Initialize the lateral connections to output zero if True
__C.FPN.ZERO_INIT_LATERAL = False

# Stride of the coarsest FPN level
# This is needed so the input can be padded properly
__C.FPN.COARSEST_STRIDE = 32

#
# FPN may be used for just RPN, just object detection, or both
#

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = False
# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0
__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to
# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5
# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = False
# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6
# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2
# FPN RPN anchor aspect ratios
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this size on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
__C.FPN.RPN_ANCHOR_START_SIZE = 32
# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False

```

其他[FPN的Pytorch实现][17]。

# Generalized RCNN结构

R-CNN包括三种主要网络：

1. Head：输入(w,h,3)生成feature map，降采样了16倍； w和h是预处理以后的图片大小哦
2. Region Proposal Network (RPN)：基于feature map，预测ROI；
    - `Crop Pooling`：从feature map中crop相应位置
3. Classification Network：对crop出的区域进行分类。

在pytorch版Detectron中，[medeling/model_builder.py][18]中的`generalized_rcnn`将FPN、fast rcnn、mask rcnn作为“插件”，通过config文件中控制“插拔”。

```
MODEL:
  TYPE: generalized_rcnn
  CONV_BODY: FPN.fpn_ResNet50_conv5_body # backbone
  FASTER_RCNN: True # RPN ON
  MASK_ON: True # 使用mask支线：mask rcnn除了box head，还有一个mask head。
```

在Detectron的实现里，可以像上面这样在config文件中灵活选择使用的backbone（比如conv body使用Res50_conv4,roi_mask_head和box head共同使用fcn_head）。代码模块划分：

```
- Conv_Body 对应下图中的head# 输入im_data，返回blob_conv
- RPN 对应下图中的Region Proposal Network: loss_rpn_cls + loss_rpn_bbox # 输入blob_conv，返回rpn_ret;
- BBOX_Branch：loss_rcnn_cls + loss_rcnn_bbox
    - Box_Head 对应下图中的Generate Grid Points Sample Feature Maps + Layer4 # 输入conv_body, rpn_ret返回box_feat
        - 通过FAST_RCNN.ROI_BOX_HEAD设置
    - Box_Outs 对应下图中的cls_score_net + bbx_pred_net# 输入box_feat, 返回cls_score, bbox_pred, 计算loss_cls, loss_bbox
- Mask_Branch: loss_rcnn_mask
    - Mask_Head# 输入blob_conv, rpn_net, 返回mask_feat
        - 通过MRCNN.ROI_MASK_HEAD设置
    - Mask_Outs# 输入mask_feat，返回mask_pred
```

![此处输入图片的描述][19]

先来看看Head怎么得到feature map。拿VGG16作为backbone来举例的话，一个完整的VGG16网络长这样：

![VGG16][20]

其中红色部分就是下采样的时刻。原始论文里使用VGG16，因为提feature map只用了最后一次max pooling前面的部分，所以留下来的四次pooling总共下采样是16倍。得到的feature map长这样：

![under sampling][21]

最终的feature映射回原图的话大概长这样：

![此处输入图片的描述][22]

有了feature map以后，开始走RCNN的主体流程：

![此处输入图片的描述][23]

## 1. Anchor Generation Layer

第一步就是生成anchor，这里的anchor亦可理解为bounding box。rpn的任务是对上上图的每个小红点都计算若干anchor（默认是9个）：

![此处输入图片的描述][24]

-  三种颜色分别代表128x128, 256x256, 512x512
-  每种颜色的三个框分别代表比例1:1, 1:2 and 2:1

> anchor是RPN的windows的大小，在feature map的每一个位置使用不同尺度和长宽比的window提取特征。原论文描述为“a pyramid of regression references”。

对应到代码上：在trainval_net.py中，imagenet的`ANCHOR_SCALES`的默认是[4, 8, 16, 32]，`ANCHOR_RATIOS`默认是[0.5,1,2]。

```
#imagenet
args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

```

但是并不是越多越好，要控制在一定的数量，理由是：

1. anchors多了就会造成faster RCNN的时间复杂度提高，anchors多的最极端情况就是overfeats中sliding windows。
2. 使用多尺度的anchors未必全部对scale-invariant property都有贡献

所以，在使用自己的数据的时候，统计一下gorund truth框的大小（注意考虑预处理rescale的系数），确定大小范围还是很有必要的。

## 2. Region Proposal Layer

先明确foreground和background的概念：前景`fg`（foreground）代表有物体（不管是哪个类别），背景`bg`（background）就是没有任何物体。

前一步生成anchor得到的是dense candidate region，rpn根据region是fg的概率来对所有region排序。

Region Proposal Layer的两个任务就是：

- `rpn_cls_score`:判断anchor是前景还是背景
- `rpn_bbox_pred`:根据**regression coefficient**调整anchor的位置、长宽，从而改进anchor，比如让它们更贴合物体边界。

![此处输入图片的描述][25]

值得注意的是，这里的anchor是以降采样16倍得到的feature map为基础的，所以总共是$\frac w {16} * \frac h {16} * 9$个anchor。每个anchor唯一对应着一个class score和bounding box regressor。

### Proposal Layer

基于fg的score，使用nms来筛除多余的anchor

![此处输入图片的描述][26]

### Anchor Target Layer

计算RPN loss：

![此处输入图片的描述][27]

- Classification Loss: cross_entropy(predicted _class, actual_class)
- Bounding Box Regression Loss:![此处输入图片的描述][28]
![此处输入图片的描述][29]
![此处输入图片的描述][30]

![此处输入图片的描述][31]

需要注意的是，fg/bg并不是“非黑即白”，而是有“don't care”这单独的一类，用来标识既不是fg也不是bg的box，这些框也就不在loss的计算范围中。同时，“don't care”也用来约束fg和bg的总数和比例，比如多余的fg随机标为“don't care”。

一些相关参数：

- `TRAIN.RPN_POSITIVE_OVERLAP`: 用来筛选fg box的阈值(Default: 0.7)
- `TRAIN.RPN_NEGATIVE_OVERLAP`: 用来筛选bg box的阈值(Default: 0.3)。这样一来，和ground truth的overlap在0.3~0.7就是“don't care”。
- `TRAIN.RPN_BATCHSIZE`: fg和bg anchor的总数 (default: 256)
- `TRAIN.RPN_FG_FRACTION`: batch size中fg的比例 (default: 0.5)。如果fg数量超过 TRAIN.RPN_BATCHSIZE$\times$ TRAIN.RPN_FG_FRACTION, 超过的部分 (根据索引随机选择) 就被标为 “don’t care”.

输入:

- RPN Network Outputs (predicted foreground/background class labels, regression coefficients)
- Anchor boxes (generated by the anchor generation layer)
- Ground truth boxes

输出：

- Good foreground/background boxes and associated class labels
- Target regression coefficients

### Proposal Target Layer

proposal layer产生ROI list，而Proposal Target Layer负责从这个list中选出可信的ROI。这些ROI经过 crop pooling从feature map中crop出相应区域，传给后面的classification
layer（head_to_tail）.



## 3.ROI Pooling Layer

简言之就是负责从feature map中提取ROI对应区域。

![此处输入图片的描述][32]

不同的pooling模式：

- crop
- align

主要用到Pytorch的**torch.nn.functional.affine_grid** 和torch.nn.functional.grid_sample

crop pooling的步骤：

1. ROI坐标 $\div$ head网络下采样的倍数（也就是stride）。需要特别指出的是：proposal target layer给出的ROI坐标是原图尺度上的（默认800$\times$600），因此映射到feature map之前要先除stride（默认是16，前面解释过）。
2.  affine transformation matrix（仿射变换矩阵）
3.  最关键的一点在于后面的分类网接收的是固定大小输入，因此这一步需要把矩形窗

![此处输入图片的描述][33]

## 4.Classification Layer

![此处输入图片的描述][34]

fc之后得到的一维特征向量被送到两个全连接网络中：

![此处输入图片的描述][35]

- `cls_score_net`：生成roi每个类别的score（softmax之后就是概率了）
- `bbox_pred_net`：结合两者得到最终的bbox坐标
    - class specific bounding box regression coefficients
    - 原来proposal target layer生成的 bbox 坐标

![此处输入图片的描述][36]

这个阶段要把不同物体的标签考虑进来了，所以是一个多分类问题。使用交叉熵计算分类Loss：

![此处输入图片的描述][220]

这个阶段依然要计算bounding box regression loss，和前面的R区别PN的区别在于：

- RPN的是为了让bbox更紧凑地贴合物体。 anchor target layer算得的target regression coefficients需要将roi box和离它最近的ground truth bbox对齐。
- classification layer中的是针对各个类别的。也就是说，对每个roi、每个类别都会生成一套coefficient。只有那些分类正确了的box才能参与，分类错误的直接不考虑了。

> 代码实现上：使用了mask array将for循环操作转化成了矩阵乘法。mask array标注了每个anchor的正确物体类别。

有意思的是，训练classification layer得到的loss也会反向传播给RPN。这是因为用来做crop pooling的ROI box坐标不仅是RPN产生的 regression coefficients应用到anchor box上的结果，其本身也是网络输出。因此，在反传的时候，误差会通过roi pooling layer传播到RPN layer。好在crop pooling在Pytorch中有内部实现，省去了计算梯度的麻烦。

# Mask-RCNN
[Mask RCNN的Pytorch实现][37]，[原始论文][38]。

mask rcnn里默认的mask head是conv层，也可以通过MRCNN.`USE_FC_OUTPUT`设置使用FC层。

- 做分割的话，关键是roi pooling时候的对齐问题，mask rcnn提出roi align
face++提出precise roi pooling，使用$x/16$而不是$[x/16]$，使用bilinear interpolation
- 分隔开分割和分类两个任务，mask rcnn中对每个类别都会生成一个mask，或者是class-agnostic的实验中，不管类别直接生成mask，效果都不错
- [关于mask rcnn实现的讨论][241]，kaggle上也有一个做医疗图像的demo

# Inference

![此处输入图片的描述][40]

# Appendix

## non-maximum suppression（nms）

![此处输入图片的描述][41]

上图左边的黑色数字代表fg的概率

- standard NMS (boxes are ranked by y coordinate of bottom right corner). This results in the box with a lower score being retained. The second figure uses modified NMS (boxes are ranked by foreground scores).

![此处输入图片的描述][42]

- This results in the box with the highest foreground score being retained, which is more desirable. In both cases, the overlap between the boxes is assumed to be higher than the NMS overlap threhold.

[讲nms（non-maximum suppression）的文章][43]

## Focal loss

看ICCV那篇focal loss的论文《Focal Loss for Dense Object Detection》.

不过这个pytorch版detectron还没实现，官方Detectron是集成在Caffe2里。可参考[Pytorch实现][44]。

$$
Loss(x, class) = - \alpha (1-softmax(x)_{[class]})^\gamma \log(softmax(x)_{[class]})
$$

- $$\alpha$$(1D Tensor, Variable) : the scalar factor for this criterion
- $$\gamma$$(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), putting more focus on hard, misclassiﬁed examples
- size_average(bool): By default, the losses are averaged over observations for each minibatch. However, if the field size_average is set to False, the losses are instead summed for each minibatch.


## 针对小物体

每个小红点之间就是16像素的间隔了，如果要检测特别细小的物体，这么大的下采样就很危险了。于是，为了尽量不破坏细小物体的清晰度，参考[github上关于检测微小物体的讨论][39]，我尝试了两种方案：

### 1. 降低下采样倍数

为了方便实验，我给网络增加了一个参数`downsample`，用来控制下采样倍数，相应地调整网络结构：

```
parser.add_argument('--downsample', dest='downsample_rate',
                  help='downsample',
                  default=16, type=int) # 原网络默认16倍
```



需要注意的是：

* 一旦改变下采样，同时还要改变feature_stride和spatial_scale。否则预测框框就变成了边缘的线条（同时loss_rpn_cls奇高）。比如VGG16（降采样8倍）：
    * __C.feat_stride': 8”
    * self.spatial_scale to (1/8)
*  vgg16尝试4倍下采样：去掉了stage5的卷积，保障输入图像最短边在1200


### 2. 切割原图

- 训练数据切patch：将原图切分为四等份再训练，保障大部分图的清晰度不被压缩。这样一来，标注也要做调整：
    - 一种方式是重新生成新的子图标注，新写一个yourdata.py；
    - 另一种偷懒方式则是修改 yourdata.py的`_load_XXX_anotation(self, index)`，使得读入每个子图，返回的也是每个子图的所有标注框。
- 测试数据为原图，沿用原来的数据读入方式即可。

## Todo List

- 加入valid过程，确认是否有过拟合:爆显存了
- negative sampling：selective search

## Reference

- [faster rcnn原始论文][46]
- 如果想了解object detection的发展史，可以看[Object Detection][47]
- 推荐阅读[Faster R-CNN: Down the rabbit hole of modern object detection][48]
- [Object Detection and Classification using R-CNNs][49]


  [1]: https://github.com/roytseng-tw/Detectron.pytorch
  [2]: https://github.com/facebookresearch/Detectron
  [3]: https://github.com/jwyang/faster-rcnn.pytorch
  [4]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/datasets/roidb.py
  [5]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/utils/boxes.py
  [6]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/roi_data/loader.py
  [7]: https://github.com/facebookresearch/Detectron/blob/e5bb3a8ff0b9caf59c76037726f49465d6b9678b/detectron/roi_data/rpn.py
  [8]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa32302afc0b.png
  [9]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/roi_data/minibatch.py
  [10]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/roi_data/fast_rcnn.py
  [11]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa46e9e0bbd7.png
  [12]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa59c8da4c4b.png
  [13]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa59d170c750.png
  [14]: https://arxiv.org/pdf/1612.03144.pdf
  [15]: http://static.zybuluo.com/sixijinling/cv2l758k7dyj022k0iinwsm1/image_1c8u6egich3e1ha41d3o1m22s3g16.png
  [16]: http://static.zybuluo.com/sixijinling/elabj076z56xa1xsfbwqyy50/image_1c8u6f7esu9i1sl510hs1pnm1qvl1j.png
  [17]: https://github.com/jwyang/fpn.pytorch
  [18]: https://github.com/roytseng-tw/Detectron.pytorch/blob/9294ec13d4a59cf449b09e1ada72a56b3420249c/lib/modeling/model_builder.py
  [19]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5a9ffec911c19.png
  [20]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/vgg.b6e48b99.png
  [21]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/image-to-feature-map.89f5aecb.png
  [22]: https://tryolabs.com/images/blog/post-images/2018-01-18-faster-rcnn/anchors-centers.141181d6.png
  [23]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa0053323ac5.png
  [24]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa05d3ecef3e.png
  [25]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa0695484e3e.png
  [26]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa5766d53b63.png
  [27]: http://www.telesens.co/wordpress/wp-content/ql-cache/quicklatex.com-142d5b70256748a64605bfc6e2f30ea9_l3.svg
  [28]: http://www.telesens.co/wordpress/wp-content/ql-cache/quicklatex.com-79e8cbe4b5682f6abc719c54d768a4ae_l3.svg
  [29]: http://www.telesens.co/wordpress/wp-content/ql-cache/quicklatex.com-f26b9d082be79d08e06cdbeb5cfc1e3a_l3.svg
  [30]: http://www.telesens.co/wordpress/wp-content/ql-cache/quicklatex.com-dae64c7ea8affa572e4b38b84688e1fd_l3.svg
  [31]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa13d4d911d3.png
  [32]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa402baba3a1.png
  [33]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa4255fdacb6.png
  [34]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa55c81eac0a.png
  [35]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa55c97f3287-1024x607.png
  [36]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa4255fdacb6.png
  [37]: https://github.com/multimodallearning/pytorch-mask-rcnn
  [38]: https://arxiv.org/pdf/1703.06870.pdf
  [39]: https://github.com/rbgirshick/py-faster-rcnn/issues/86
  [40]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa70ff399c57.png
  [41]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa7c84451f81.png
  [42]: http://www.telesens.co/wordpress/wp-content/uploads/2018/03/img_5aa7c828703ab.png
  [43]: https://zhuanlan.zhihu.com/p/31427728
  [44]: https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
  [46]: https://arxiv.org/pdf/1506.01497.pdf
  [47]: https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/
  [48]: https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
  [49]: http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/