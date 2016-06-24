---
layout: post
title: NN-based APP评论标记
category: project
description: 这是本科毕设，通过对APP评论进行分类，提取bug report、功能需求之类的信息，事实上就是一个多分类问题。
---

# 1. 背景

APP评论中包含大量价值信息，譬如bug report能帮助开发者决定新版本的修复，user experience能帮助用户决定是否下载该应用。由于评论数量庞大，采取自动化方法进行分类则能省去大量人工成本，为需求方提供其更关注的信息。然而，实际采样发现一条评论中可能包含不止一种事件类型，我们的目标也就变成解决短文本的多标签分类问题。此外，评论的长度波动也较大（短的只有一个词，长的达到三百多词），又为这个任务增加了难度。

首先，我们通过词向量来表征评论文本，作为学习模型的输入，进而使用神经网络模型在训练集上进行学习。我们分别实现了卷积神经网络模型（CNN）和门限循环网络模型（GRU），输出层的输出即为某评论被分为不同标签的概率，可人为设定阈值来判断是否赋予该标签。最后，我们通过在测试集上进行实验评估并比较其最终分类效果。

从代码来说明：

```
    self.labels = labels
    self.word2vec_model = word2vec_model
    self.scaler = scaler
    self.keras_model = keras_model
```

- labels：标签集合；
- word2vec_model：使用word2vec训练得到的词向量模型，用于构造分类器的输入；
- scaler：对作为输入的词向量进行标准化；
- keras_model：基于keras构造学习网络。

接下来按处理流程顺序介绍。

# 2. 数据集 

使用的数据集是[2015-training-set.sql][1]， 其中包括从 Apple 和 Google Play app stores 采集的评论，并按主题人工标记了四个标签：

- **User Experience**
    - "I use this app almost every weekend while exploring back roads and trails by motorcycle. Functionality is excellent while on the road as well as using the data to review later using Google Earth"
- **Bug report**
    - "After the new update, my mobile freezes after I've been using the app for a few minutes"
- **Feature/Improvement request**
    - "I wish you could add a link that would allow me to share the information with my Facebook friends”
- **Rating**
    - “I cannot believe how amazing the new changes are.”
    
![2016-04-13 20-20-16屏幕截图.png-417.5kB][2]

多标签评论举例：

> “Dosen’t work at the moment.Was quite satisfied before the last update.Will change the rating once it’s functional again”

应被归类为故障报告和感性评价，而不是用户体验或功能需求。
评论

> “Wish it had live audio, like Voxer does. It would be nice if you could press the play button and have it play all messages, instead of having to play for each individual message. Be nice to have a speed button, like Voxer does, to speed up the playing of messages. Be nice to have changeable backgrounds, for each chat, like We Chat does. Other than those feature requests, it’s a great app ”

应被分类为功能需求以及评价，而不是故障报告或用户体验。

评论的长度分布如下图：

![review.png-24.2kB][3]

# 3. 构造输入

## 3.1 word representation

我们的第一步是得到高质量的词表示(Word Representation)，即将词进行数值化，
同时保留其语义、语法层面的近邻关系。继而用这些词表示来表征评论文本，作为机器学习模型的输入。在这一步使用[Word2vec][4]。
![word2vec.png-427.4kB][5]
本文使用**CBOW**来从大量无标签文本中学习词的分布式表示。为学习得到高质量的词向量，我们首先需要庞大的英文语料。维基百科官方提供了一个很好的[维基百科数据源][6]，可以方便的下载多种语言多种格式的维基百科数据。我们选择在在的英文维基语料上训练，下载得到.处理包括两个阶段，首先将xml的wiki数据转换为text格式。本文中使用的机器是intel i5，RAM 8G。在运行三个小时后，得到了13.6G的wiki.en.txt。每篇文章一行，且忽略掉标点符号。其词典足以覆盖评论文本中出现的词汇。我们在维基语料上训练得到了100维的词向量作为后续神经网络模型的输入。

CBOW模型结构如下图，包含三层：输入层、投影层和输出层。

![cbow.gif-5.3kB][7]

- **输入层**：包含当前词$$w$$的上下文$$Context(w)$$中$2c$个词的词向量$$v(Context(w)_1),v(Context(w)_2),\dots,v(Context(w)_{2c}) \in \mathbb {R}^m$$，这里$$m$$即为词向量长度；
- **投影层**：将输入层的$$2c$$个向量做求和累加，即$$x_w=\sum_{i=1}^{2c}v(Context(w)_i) \in \mathbb {R}^m$$；
- **输出层**：输出层对应一棵二叉树——一棵以语料中出现过的词当叶子节点、以各词在语料中出现的次数当权值构造出来的Huffman树。在这棵Huffman树中，叶子节点共$$N$$个，分别对应词典$$\mathcal D$$中的词，非叶子节点$N-1$个。

sigmoid函数是神经网络中常用的激活函数之一，其定义为

$$
\sigma(x)=\frac 1{1+e^{-x}}
$$

利用sigmoid函数，对于任意样本$$x=(x_1,x_2,\dots ,x_n)^\mathrm{T}$$，可将二分类问题的预测函数写为

$$
h_{\theta}(x)=\sigma (\theta ^\mathrm{T}x)
$$

取阈值$$T=0.5$$，则二分类的判别公式为

$$
y(x)=\cases
{ 
{1}&$h_{\theta}(x)\geq 0.5$\\
{0}&$h_{\theta}(x)< 0.5$
}
$$

确定一个整体损失函数，然后对其进行优化，从而得到最优的参数$$\theta$$。

需要利用$$x_w \in \mathbb {R}^m$$以及**Huffman树**来定义函数$$p(w|Context(w))$$。Huffman树的每一次分支都可视为进行了一次二分类，这样就需要为每一个非叶子节点的左右孩子结点指定一个取值为0或1的Huffman编码。word2vec将编码为0的结点定义为正类，编码为1的点定义为负类。根据二分类的预测函数，一个结点被分为正类的概率为：

$$
\sigma (x_w^\mathrm{T} \theta)=\frac 1{1+e^{-x_w^\mathrm{T} \theta}}
$$

则被分为负类的概率则为$$1-\sigma (x_w^\mathrm{T} \theta)$$，这里Huffman树每个非叶子结点对应的向量则为我们要优化的参数$$\theta$$。对于词典$$\mathcal D$$中的任意词$$w$$，Huffman树中必存在一条从根结点到词$$w$$对应结点的路径$$p^w$$（且这条路径是唯一的）。

对于CBOW模型，word2vec给出了两套框架，它们分别基于Hierarchical Softmax和Negative sampling来进行设计。我们选择使用基于**Negative sampling**的CBOW模型。

Negative sampling（简称NEG）是Noise Contrastive Estimation的一个简化版本，目的是用来提高训练速度并改善所得词向量的质量。NEG利用随机负采样来大幅度提高性能。在CBOW模型中，已知词$$w$$的上下文$$Context(w)$$，需要预测$$w$$。对于给定的$$Context(w)$$，词$$w$$就是一个**正样本**，其他词就是**负样本**了。

假定已选好一个关于$$w$$的负样本子集$$NEG(w)\notin \not 0$$，且对$$\forall \tilde{w} \in \mathcal D$$，定义

$$
L^w(\tilde w)=\cases
{ 
{1}&$\tilde w=w$\\
{0}&$\tilde w \neq w$
}
$$

表示词$$\tilde w$$的标签，即正样本的标签为1，负样本的标签为0。

路径$$p^w$$生存在$$l^w-1$$个分支，将每个分支看做一次二分类，每一次分类就产生一个概率，将这些概率乘起来，就是所需的$$p(w|Context(w))$$。

$$
p(u|Context(w))=[\sigma (x_w^\mathrm{T} \theta ^u)]^{L^w(u)}\dot [1-\sigma (x_w^\mathrm{T}\theta ^u)]^{1-L^w(u)}
$$

对于一个给定的正样本$$(Context(w),w)$$，我们希望最大化

$$
g(w)=\prod _{u \in \{w\}\cup NEG(w)}p(u|Context(w))
$$

这里$$x_w$$仍表示$$Context(w)$$中各词的词向量之和，而$$\theta ^u \in \mathbb {R}^m$$表示词$$u$$对应的一个向量，为待训练参数。

从形式上看，最大化$$g(w)$$相当于最大化$$\sigma (x_w^\mathrm{T} \theta ^w)$$，同时最小化所有的$$\sigma (x_w^\mathrm{T} \theta ^u)$$，$$u \in NEG(w)$$。增大正样本的概率同时降低负样本的概率。于是，对于一个给定的语料库$$\mathcal C$$，函数

$$
G=\prod _{w \in \mathcal C}g(w)
$$

即为整体优化目标。为计算方便，对$$G$$取对数。最终的目标函数记为$$\mathcal L$$

接下来利用随机梯度上升法进行优化，其做法是：每取一个样本$$(w,Context(w))$$，就对目标函数中的所有（相关）参数做一次刷新。

![ScreenShot_20160620012252.png-33.3kB][8]

同样的词向量生成算法还有[GloVe][9]。
    

## 2. 文本预处理

使用通用的NLP技术对训练数据进行预处理操作能帮助提高分类正确率，包括去停用词、词干化、词形还原等。

在进行预处理之前，首先使用**NLTK**的 `punkt sentence segmenter`对评论文本进行分句。之后要把每句话再逐个切割成单词，最简单的方法是使用NLTK包中的 `WordPunct tokenizer`。

**停用词**指的是常用且被视为没有价值的单词，如“the”、“am”这种并不影响评论语义的词。将它们去除掉能降低噪声，强化“bug”“add”这类有价值的信息的影响力。然而，一些被定义为停用词的关键词也可能是和评论分类相关的：比如“should”和“must”可能指示的是功能需求，而“did”“while”“because”指示的是功能描述。

词元（lemma）指的是具有相同词干、主要词性及相似词义的词汇集合，例如“cat”和“cats”就是同一个词元。词形（Wordform）指的是词在形式上的曲折变化。

形符化是将不同词形还原到词元本身的过程，例如“fixing”“fixed”“fixes”都来源于“fix”。类似地，归一化指的是去除单词后缀来还原词的基础形式，如“window”“Window”“windows”都对应“window”。词形还原考虑语言学上下文并使用词典，而词干提取只是对单个的词操作，无法根据词性区分词的不同涵义。比如词形还原中会将“better”还原为“good”，而词干提取会把“goods”和“good”一视同仁。词形还原和词干提取都能帮助统一具有意义相同、词形不同的关键词，从而缩小词典大小。我们使用NLTK的PorterStemmer从英文单词中获得符合语法的（前缀）词干。
 
## 3. 标准化（scaler）

在将词向量输入机器学习模型之前，我们首先要进行正则化。原因是几乎所有的机器学习算法都基于梯度方法，而大部分梯度方法对数据的尺度高度敏感，因此在我们运行机器学习算法之前，我们应该进行正则化（normalization）,或者所谓的标准化（standardization）。

所谓标准化，就是指特征都符合标准正态分布（高斯分布：均值是0、方差是1），即所有的特征都是标准化的，防止某个特征权重过大影响结果。考虑到训练集样本可能很庞大，选择分批（batch）计算处理，设定批大小（batchsize）为1024。使用\citet{chan}提出的等式1.5a，b来计算增量式的均值和标准差。Algorithm 2展示了标准化的过程：对所有出现在评论训练集中的词进行标准化。

![ScreenShot_20160620012347.png-46.2kB][10]

# 4. 学习模型

在对词向量计算均值和标准差后，我们需要构造模型的输入和预期输出。为了进行批训练，需要对评论文本进行零补足或截断，从而确保所有文本都是固定长度$$N$$。

这里我们固定将所有评论截断至50个词，即用这50个词对应的词向量来表示一条评论。对训练集和测试集的所有评论，我们都对它构造一个$$n$$行$$k$$列的矩阵，其中$$n$$是单词个数，这里是50，$$k$$是词向量的维数，这里是100。矩阵中的第$$i$$行即代表文本中的第$$i$$个词的词向量。这里使用标准化步骤中计算的词向量均值和标准差来对所有评论文本词向量进行标准化，即将词向量的每一维（按列进行）减去其均值，并除以其方差。此外，我们将评论对应的人工标记标签作为预期输出，为学习模型构造一个$$s$$行$$l$$列的目标输出矩阵，其中$$s$$是样本个数，分别对应训练集和测试集中的评论条数，$$l$$是标签个数，这里是4，在对应的标签位置使用bool值来表示某评论是否具有该标签。

训练神经网络是找到合适权重和偏置使得计算输出更接近训练样本对应的已知结果的过程。一旦找到这些参数值，就能利用当前的模型预测未知数据对应的输出。而神经网络训练有两种常用的方法：在线训练和批训练。这两种训练方式能产生不同的结果。通常研究者们的共识是在使用BP训练时，使用在线方法要优于批训练。

事实上，在线训练可理解为批训练一次参数更新的迭代只使用一个样本，而批训练则是多个样本通过后才更新。本文选择使用批训练，即在每一次迭代中，根据设置的批大小一次性通过若干个训练样本，全部通过完毕后在验证集上进行测试。每一次选取批大小个样本都会先打散样本。

## 4.1 CNN

CNN最初被应用于图像分类和图像检测，并取得成功应用，后来被引入到许多CNN任务中，用于提取文本特征。CNN的核心在于卷积核，或者叫做滤波器（filter），从信号处理的角度而言，滤波器是对信号做频率筛选，这里主要是空间-频率的转换，CNN的训练就是找到最好的滤波器使得滤波后的信号更容易分类，还可以从模版匹配的角度看卷积，每个卷积核都可以看成一个特征模版，训练就是为了找到最适合分类的特征模版。

在构造完神经网络的输入、输出矩阵后，我们对每个评论文本所对应的矩阵进行卷积操作：对$$h$$行$$k$$列的窗口都应用一个卷积核矩阵$$W\in \mathbb R^{hk}$$（后文我们记作filter）,其中$$h$$是n-gram模型的长度，即窗口长度，从而计算出一个标量$$c_i$$，可理解为该窗口范围内抽象出的上一层特征。计算公式为：

$$
c_i=f(W_{x_{i:i+h-1}}+b^{(c)})
$$

其中$$b$$是一个偏置量，而$$f$$是我们选择的非线性激活函数。在整个卷积过程中，我们将该卷积核应用于所有$$N-h+1$$个窗口，得到了一个特征集合$$\{c_1,c_2,\dots,c_{N-h+1}\}$$。在这之后，对这$$N-h+1$$个特征应用一个 max-over-time的采样操作，继而得到特征$$\hat c$$。

$$
\hat c=\max(c_1,c_2,\dots ,c_{N-h+1})
$$

这样一来，我们就从卷积核产生的特征集合中抽取了单个特征。接下来对许多$$h$$行$$k$$列的feature maps以及窗口大小$$h$$不同的filter重复上述过程。最后将所有这些特征串联成一个特征向量，即向量$$c'$$。
最后加一个大小为$$|L|$$、激活函数为sigmoid的全连接输出层,用于计算每一个可能标签的概率。

$$
\hat y= \sigma(Uc'+b^{(o)})
$$

在训练过程中，这些概率被用来计算误差。而在测试过程中，我们根据一个设定好的阈值来判断每个标签赋值0或1.

本文使用的CNN模型如下图所示，共分四层，第一层是词向量层，即我们为每个评论文本构造的$$n$$行$$k$$列矩阵；第二层是卷积层，多个filter作用于词向量层，不同filter生成不同的`feature map`；第三层是采样层，取每个feature map的最大值；第四层是一个全连接的softmax层，输出是每个标签的概率。除此之外，输入层可以有两个channel，其中一个channel采用预先利用word2vec训练好的词向量，另一个channel的词向量可以通过BP在训练过程中调整。而这里使用的是静态的词向量。

![cnn_model.png-26.8kB][11]

## 4.2 GRU

和RNN相似，GRU使用权重集合来构造隐藏层向量$$h$$。之后在一段时间序列中，使用相同的权重来结合$t$时刻的输入和前一刻的隐层向量$$h_{t-1}$$。然而，不同于标准的RNN，GRU增加了两个“门”——一个更新门$$z$$和一个重置门$$r$$：

$$
z=\sigma(W_zx_t+U_zh_{t-1}+b_z)\\
r=\sigma(W_rx_t+U_rh_{t-1}+b)
$$

通过重置门来决定前一个状态需要传递多少到记忆单元，该过程定义为：

$$
\tilde h=\tanh(Wx_t+r\odot Uh_{t-1}+b_r)
$$

最后，我们使用更新门来计算前一个隐层状态和$$t$$时刻记忆单元的加权平均：

$$
h_t=z_t \odot h_{t-1}+(1-z_t)\odot \tilde h
$$

通过这些门，模型有能力学习到何时该忘记，或是重新训练关于序列的特定信息。在本文提出的模型中，将GRU应用于文本中，词向量作为$$t$$时刻的输入，产生最终的隐层输出层$$h_f$$。最后将这个向量输入一个全连接层，输出层大小为$$|L|$$，来产生最后的输出$$\hat y$$，其中第$$i$$个元素代表标签$$i$$被赋予当前文本的概率。

$$
\hat y=\sigma(W^{(o)}h_t+b^{(o)})
$$

和CNN类似，训练过程中我们使用$$\hat y$$来计算模型误差。测试过程中，通过阈值$$t$$判断概率为0或1，从而确定文本是否应被贴上标签$$i$$。

GRU的结构如下图所示。GRU首先根据当前输入的词向量以及前一个隐藏层的状态（hidden state）计算出更新门和重置门。再根据重置门、当前词向量以及前一个隐藏层的状态计算新的记忆单元内容(new memory content)。当重置门为1的时候，新的记忆单元内容忽略之前的所有记忆单元内容，最终的记忆是之前的隐藏层的状态与新的记忆单元内容的结合。 

![gru.png-38.2kB][12]

# 实验

![cnn.png-43.9kB][13]

![models1.png-86.1kB][14]


  [1]: https://mobis.informatik.uni-hamburg.de/app-review-analysis/
  [2]: http://static.zybuluo.com/sixijinling/33igbmlcaj4vmz5fcf9wsdho/2016-04-13%2020-20-16%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [3]: http://static.zybuluo.com/sixijinling/tibv12f4ku9f28qakqbz911o/review.png
  [4]: https://code.google.com/p/word2vec/
  [5]: http://static.zybuluo.com/sixijinling/7aud7dormsavts73ze33ws4h/word2vec.png
  [6]: https://dumps.wikimedia.org
  [7]: http://static.zybuluo.com/sixijinling/e8c8mntqrzp4gl33f3h64fp3/cbow.gif
  [8]: http://static.zybuluo.com/sixijinling/95dxwu000l861fkwu9cnpel7/ScreenShot_20160620012252.png
  [9]: http://nlp.stanford.edu/projects/glove/
  [10]: http://static.zybuluo.com/sixijinling/95sw4bhfhy44y9dalop95c8e/ScreenShot_20160620012347.png
  [11]: http://static.zybuluo.com/sixijinling/3pvl8r8yrunkkj8g1k37yqc5/cnn_model.png
  [12]: http://static.zybuluo.com/sixijinling/w2rsavt0a972gl11p6i6c2oy/gru.png
  [13]: http://static.zybuluo.com/sixijinling/491pqhyovpf7xgbn0iqy9e0a/cnn.png
  [14]: http://static.zybuluo.com/sixijinling/2dvpfegbl59zzffix3nw09zc/models1.png
