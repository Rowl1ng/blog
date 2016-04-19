---
layout: post
title: NN-based APP评论标记
category: project
description: 这是本科毕设，通过对APP评论进行分类，提取bug report、function requirements之类的信息，事实上就是一个多分类问题。
---

可怜的毕设哟

- 第一部分：开题报告
    - [研究背景](#ID1)
    - [研究内容](#ID2) 
    - [研究方法](#ID3) 
    - [研究步骤](#ID4) 
    - [参考文献](#ID5)

- 第二部分：中期报告
    - [Data Collection](#ID6)
    - [中期报告](#ID6)
    - [中期报告](#ID6)
    - [中期报告](#ID6)

## 一.研究背景 {#ID1}

随着数以千计的开发者、应用不断涌入，移动app市场的应用数量持续大幅度增涨。当前的移动应用市场不仅仅是Android App Store，还有Apple App Store，Blackberry World Store，Microsoft APP Store这些全球市场以及区域化的应用商店。除了充当用户下载app的渠道，这些APP商店还会通过app用户评论反馈用户体验。
这些用户评论（这里以及后文中的的用户评论指的是对app描述性的部分）往往潜藏着巨大信息量，譬如用户的新功能需求、对某项功能的不满以及安全隐患。而这些信息对三方————开发者、用户和APP商店经营者来说，都是极其有价值的：

 1. 开发者可以得到对所开发app的及时反馈，比如bug、新需求以及其他信息；The first scenario is useful to developers and users when comparing issues across competing apps.
 2. 用户能通过评论信息决定是否下载或购买该应用；
 3. 对于Apple、Blackberry、Google和Microsoft这些商店经营方来说，也可以通过分析用户评论挖掘异常的app,或是app之间的关联。The second scenario is useful to app store owners and developers as we analyze the issue distribution of apps in the Google Play Store by app category (e.g., social, finance) and compare the issues from competing app stores for the same app titles.The third scenario is useful to app store owners when detecting anomalous apps based on user reviews and finding apps that violate or disregard the policies and guidelines of an app store (i.e., Google
Play Store).
 
由于这些评论数量庞大且格式自由，仅单纯地依赖人工审查来从中提取信息是不切实际的。对于热门app来说，甚至每天都会有数百条新评论。
事实上，对评论信息的挖掘已经带动了一系列移动应用分析公司的诞生，这些公司旨在为客户（应用开发者）提供专业化的用户评论数据和对比分析，比如Flurry和App Annie。
However, much of the provided analytics are not
software engineering oriented yet. For example, the occurrence of words in reviews across
competing apps are presented, however, the provided analysis would not link such word
occurrences to software related concepts (e.g., software quality). 
自动化方法可基于评论所提到的事件类型（功能需求，不满，隐私泄露）自动标记评论。然而用户可能在一条评论中提及多类事件，因此不能用单一类别来标记这样的评论。
 However, users
might raise several issues within a particular review. Figure 1 shows an excerpt of a user
review where the user raises three issues about an update issue, the response time, and a
functional complaint. We cannot label such user reviews with only a single issue. Moreover,
such labelling is a difficult task due to the unstructured nature of reviews with many of them
containing slang, lacking punctuation, and containing improper grammar.

## 二. 研究内容  {#ID2}
本毕业设计中旨在开发对用户评论自动化多分类方法，以帮助相关方获得对评论中的用户反馈的全面认识，同时提供对应用商店的全局认识以及异常应用检测。
研究目标包括分析多分类用户评论的类型范围the extent of multi-labelled user reviews,评估自动标记多类别评论的效果，以及说明自动分类用户评论对不同利益相关方的应用。
关注点对在于多分类机制的表现的评估方法，研究对象是Google Play Store中的用户评论。需要补充的是，这些评论伴天然伴随着满分为五星的星级打分，这点有利于研究过程中的情感分析。

## 三. 研究方法  {#ID3}
基于普及程度和评论收集工具的可用性，我们选择了当前的主流APP商店————Google Play Store。根据Bostic调查，在2013年7月Google Play Store就已拥有了超过一百万个应用。

## 四. 研究步骤  {#ID4}
第一阶段（评论采集）：编写一个简单的web爬虫从APP商店中自动采集用户评论。主要选择一星和两星的用户评价，因为它们更侧重于负面情况，这也正是用户关注的核心内容。

第二阶段（情感分析）：编写工具对所研究的App的负面评价（一星或两星）、中性评价（三星）、正面评价（四星或五星）进行情感分析。该工具预期达到对负面、正面评论分类的较高准确率。
第三阶段（人工标记）：这个阶段需通过人工审查来识别应用评论的不同事件类型。分析的目标是提炼出共有的概念，而选择这些概念的依据是其对开发者的价值且是独立于特定app本身特性的，得到的主题集合作为未来工作的基准（这样相关方就可以根据他们的需求来选择不同的主题子集，因为整个机制是独立于所选择的事件类型的）。事实上，这一部分的工作成果将为下一阶段的自动标记提供可靠的训练集。
In total, we identify 13 issue types from
the user reviews that we randomly sampled as shown in Table 4. We include a 14th
issue type (“other”) that covers any user review that does not conform to the 13 issue
types.
第四阶段（自动标记）：
1.预处理阶段，输入是上阶段人工标记后的评论数据，在进行一系列预处理操作（如去除停用词、过滤数字和特殊字符）后将大大简化后续处理的复杂度。将处理后的文本转换成机器学习算法的可操作输入，即将文字转换为数字矩阵（映射关系暂不详述）；
2.建立合适的机器学习模型，输入为文本对应的数字矩阵，对训练集进行迭代训练，采取一定措施优化参数，得到标记结果；
3.评估模型有效性，这一阶段主要采用交叉验证、F指数等手段或标准来验证模型的分类正确性，同时针对性提出可能的改进方案。
第五阶段（总结）：整理实验数据，最终调试程序，制作图表，撰写并修缮论文。
## 五. 参考文献  {#ID5}
[1] Analyzing and automatically labelling the types of user
issues that are raised in mobile app reviews,Stuart McIlroy , Nasir Ali, Hammad Khalid, Ahmed E. Hassan,Springer Science+Business Media New York 2015
[2] Bug Report, Feature Request, or Simply Praise?On Automatically Classifying App Reviews,Walid Maalej,Hadeer Nabil,Arxiv 2015


---

# 中期检查
---

## 1. Data Collection  {#ID6}

![2016-04-13 20-20-16屏幕截图.png-417.5kB][1]

- 数据集：
    - [2015-training-set.sql][2]：
        - user reviews from the Apple and Google Play app stores 
    - 分类(多分类)：
        - User Experience
            - "I use this app almost every weekend while exploring back roads and trails by motorcycle. Functionality is excellent while on the road as well as using the data to review later using Google Earth"
        - Bug report
            - "After the new update, my mobile freezes after I've been using the app for a few minutes"
        - Feature/Improvement request：
            - "I wish you could add a link that would allow me to share the information with my Facebook friends”
        - Rating
            - “I cannot believe how amazing the new changes are.”
    - 连接MySQL：
    
    ```python
        db = pymysql.connect(host='localhost', port=3306,user='root',passwd='2005726',db='review')
        cur = db.cursor()
        self.db = db
        self.cur = cur
        self.fetch_data()
    ```
    
    - 读取数据：
    
    ```python
        train = []
        self.cur.execute("SELECT * FROM Bug_Report_Data")
        for row in self.cur:
            review = str(row[16])
            review= str(review.decode('utf-8', errors='ignore'))
            train.append((review, 'bug'))
    ```
    
    - label输出方式：
        - To train the model you need to have a large corpus of labeled data in a text format. Magpie looks for .txt files containing the text to predict on and corresponding .lab files with assigned labels in separate lines. A pair of files containing the labels and the text should have the same name and differ only in extension：
    
    ```
            $ ls training-directory
        100.txt  100.lab  101.txt  101.lab  102.txt  102.lab  ...
    ```

## word representation

magpiemodel

        self.keras_model = keras_model
        self.word2vec_model = word2vec_model
        self.scaler = scaler
        self.labels = labels
    
### 1. Train

train_word2vec(train_dir, vec_dim=vec_dim)

```python    
       def train(self, train_dir, vocabulary, test_dir=None, callbacks=None, nn_model=NN_ARCHITECTURE, batch_size=BATCH_SIZE,nb_epochs=NB_EPOCHS, verbose=1):
            
```

Train the model on given data：

- param train_dir: directory with data files. Text files should end with
'.txt' and corresponding files containing labels should end 
- param vocabulary: iterable containing all considered labels
- param test_dir: directory with test files. They will be used to evaluate the model after every epoch of training.
- param callbacks: objects passed to the Keras fit function as callbacks
- param nn_model: string defining the **NN architecture** e.g. 'crnn'
- param batch_size: size of one batch
- param nb_epochs: number of epochs to train
- param verbose: 0, 1 or 2. As in Keras.
- return: History object


### word2vec

  Initialize the model from an iterable of `sentences`. Each sentence is a list of words (unicode strings) that will be used for training.
The `sentences` iterable can be simply a list, but for larger corpora,consider an iterable that streams the sentences directly from disk/network.
See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in this module for such examples.
If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it in some other way.

- `sg` defines the **training algorithm**. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
- `size` is the dimensionality of the feature vectors.
- `window` is the maximum distance between the current and predicted word within a sentence.
- `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
- `seed` = for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed).
- `min_count` = ignore all words with total frequency lower than this.
- `max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to `None` for no limit (default).
- `sample` = threshold for configuring which higher-frequency words are randomly downsampled; default is 1e-3, useful range is (0, 1e-5).
- `workers` = use this many worker threads to train the model (=faster training with multicore machines).
- `hs` = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
- `negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
- `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean. Only applies when cbow is used.
- `hashfxn` = hash function to use to randomly **initialize** weights, for increased training reproducibility. Default is Python's rudimentary built in hash function.
- `iter` = number of iterations (epochs) over the corpus.
- `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and returns either `util.RULE_DISCARD`, `util.RULE_KEEP` or `util.RULE_DEFAULT`. Note: The rule, if given, is only used prune vocabulary during `build_vocab()` and is not stored as part of the model.
- `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before  assigning word indexes.
- `batch_words` = target size (in words) for batches of examples passed to worker threads (and thus cython routines). Default is 10000. (Larger batches can be passed if individual texts are longer, but the cython code may truncate.)

### 1. Build Vocabulary

```python
self.scan_vocab(sentences, trim_rule=trim_rule)  # initial survey
self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
self.finalize_vocab()  # build tables & arrays
```

Build vocabulary from a sequence of sentences (can be a once-only generator stream). Each sentence must be a list of **unicode** strings.
Create a binary **Huffman tree** using stored vocabulary word counts. Frequent words will have shorter binary codes. Called internally from `build_vocab()`

```python
def fit_scaler(data_dir, word2vec_model=WORD2VEC_MODELPATH, batch_size=1024,
               persist_to_path=None):
    """ Get all the word2vec vectors in a 2D matrix and fit the scaler on it.
     This scaler can be used afterwards for normalizing feature matrices. """
```

```python
    while not no_more_samples:
        batch = []
        for i in xrange(batch_size):
            try:
                batch.append(doc_generator.next())
            except StopIteration:
                no_more_samples = True
                break
```
## 
- 训练 word vector :
    - 使用评论训练词向量模型 ，Before you train the model, you need to build appropriate word vector representations for your corpus. In theory, you can train them on a different corpus or reuse already trained ones (tutorial).
    
    ```
    >>> from magpie import MagpieModel
    >>> model = MagpieModel()
    >>> model.train_word2vec('/path/to/training-directory', vec_dim=100)
    ```

#### 参数	

    >>> model = Word2Vec(sentences, min_count=10)  # default value is 5
    
    >>> model = Word2Vec(sentences, size=200)  # default value is 100
    
    >>> model = Word2Vec(sentences, workers=4) # default = 1 worker = no parallelization

#### 初始化matrix

```python
self.vocab = {}  # mapping from a word (string) to a Vocab object
self.index2word = []  # map from a word's matrix index (int) to word (string)
```

选取一个seed string，对vocabulary中的word逐个初始化random vector。

```python
once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
return (once.rand(self.vector_size) - 0.5) / self.vector_size
```
### 训练

Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

To support linear learning-rate decay from (initial) alpha to min_alpha, either `total_examples` (count of sentences) or `total_words` (count of raw words in sentences) should be provided, unless the  sentences are the same as those that were used to initially build the vocabulary.


train batch sg:
Update skip-gram model by training on a sequence of sentences.  Each sentence is a list of string tokens, which are looked up in the model's
 vocab dictionary. Called internally from `Word2Vec.train()`.
This is the non-optimized, Python version. If you have cython installed, gensim will use the optimized version from word2vec_inner instead.
   
- 模型：
    - 词向量：200维 word2vec 在数据集上训练
    - 卷积神经网络：
        - filter：LeCun uniform initialization
        - 激活函数：tanh
        - dropout rate＝0.5
    - train
        - tokens＝300：zero－pad＋truncate
        - Max－over－time pooling
        - batch training
        - Adadelta：
    - library：
        - theano
        - Keras

## Predict

 Predict labels for a given Document object
- param doc: Document object
- return: list of labels with corresponding confidence intervals

        
```python
def _predict(self, doc):
       
```

## Multi-Label
### Binary Relevance (BR) 


## 工具类

### 1.Document.py

```python
def get_all_words(self):
        """ Return all words tokenized, in lowercase and without punctuation """
        return [w.lower() for w in word_tokenize(self.text)
                if w not in PUNCTUATION]
```

利用nltk的tokenize:

```python
def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
```

- 评价指标：
    - micro－F1指数

-----


  [1]: http://static.zybuluo.com/sixijinling/33igbmlcaj4vmz5fcf9wsdho/2016-04-13%2020-20-16%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [2]: https://mobis.informatik.uni-hamburg.de/app-review-analysis/


## 写论文

***To do list***
-[ ] 在linux下还是windows下用latex呢？


[1]：https://mobis.informatik.uni-hamburg.de/app-review-analysis/
