---
layout: post
title: 探索AlphaGo
category: project
description: 还不怎么会下围棋呢。
---
# [{{ page.title }}][1]
2016-03-15 By {{ site.author_info }}
# Alphago探索

标签（空格分隔）： 深度学习 围棋

---

开源
缩减

 - 广度：减少需要模拟的落子选项 
    - imitating expert moves(supervised learning)
    - 强化学习（self-plays）

、

 - 深度：Board Evaluation



## 环境

### Cython==0.23.4

Cython的主要目的是： 简化python调用c语言程序的繁琐封装过程，提高python代码执行速度（C语言的执行速度比python快）

    sudo apt-get install cython
### scipy==0.17.0 numpy==1.10.4

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

### h5py==2.5.0
h5py：将数据储存在hdf5文件中。

    sudo apt-get install libhdf5-dev
    sudo apt-get install python-h5py
### PyYAML==3.11
 YAML is a data serialization format designed for human readability and interaction with scripting languages.

    pip install pyyaml
### six==1.10.0
Six is a Python 2 and 3 compatibility library. It provides utility functions for smoothing over the differences between the Python versions with the goal of writing Python code that is compatible on both Python versions. 
    pip install six

### wheel==0.29.0

    sudo apt-get install python-pip
    pip install wheel
### 安装Theano和keras
-e git://github.com/Theano/Theano.git@eab9cf5d594bac251df57885509394d2c52ccd1a#egg=Theano
    git clone git://github.com/Theano/Theano.git
    cd Theano
    python setup.py develop --user
    cd ..

执行之后，将Theano目录下的theano目录拷贝到python安装目录下的dist-package下就可以了，我的机器是/usr/lib/python2.7/dist-packages

#### 安装Keras==0.3.2
这就没什么好说的了，自己下载下来就行了，keras Github地址（https://github.com/fchollet/keras）。

    python setup.py develop --user

### 插播：清理Ubuntu
清理下载的缓存包：

    sudo apt-get autoclean
    sudo apt-get clean
清理不再需要的包：

    sudo apt-get autoremove

## 开始下棋辣
今天（2016.3.16）看纪录片《围棋》第五集才知道"气"的英文是“liberty”

`go.py`基本的设置
    WHITE = -1
    BLACK = +1
    EMPTY = 0
    PASS_MOVE = None
    
`self.liberty_sets` is a 2D array with the same indexes as `board` each entry points to a set of tuples - the liberties of a stone's connected block. By caching liberties in this way, we can directly optimize update functions (e.g. do_move) and in doing so indirectly speed up any function that queries liberties
### 界面
` /interface/server`
### Alphgo/ 
#### models/


`preprocessing.py`
a class to convert from AlphaGo GameState objects to tensors of **one-hot**
	features for NN inputs
`game_converter.py`

[BeiYuu]:    http://rowl1ng.com  "Rowl1ng"
[1]:    {{ page.url}}  ({{ page.title }})
