---
layout: post
title: 概率论与随机过程——说人话
description: 还是蛮佩服听不懂但坚持上课摸鱼的我。。。
category: blog
---

## 第一章 概率空间


    
### 集代数

- 包括$\varOmega$ 
- 对余封闭
- 对**有限**并封闭

的非空**集合类**。

### $\sigma$-代数

性质：

- 一定是集代数
- 对**可数**并封闭 $\Longrightarrow$ 对**可数**交封闭

- $\mathscr F _{\varOmega}$：由$\varOmega$的所有子集组成的集合类 $\Longrightarrow$ 是$\sigma$-代数
- $\mathscr G$：由$\varOmega$的一些**集合**组成的非空**集合类** $\Longrightarrow \mathscr G \subset \mathscr F _{\varOmega}$

包含某一集合类$\mathscr G$的最小$\sigma$-代数：
$\mathscr F _{0}$即是**所有**包含$\mathscr G$的$\sigma$-代数的交，这样的$\mathscr F _{0}$记为$\sigma(\mathscr G)$

### Borel域

$$
\mathscr G=\{(-\infty,a]:a\in \mathbf R^{(1)}\}
$$

则：

- $\mathscr B^{(1)} $：$\sigma(\mathscr G)$为$\mathbf R^{(1)}$上的**Borel域**。
- $\mathscr B^{(1)} $中的集：**Borel集**。

扩展至$\mathbf R^{(n)}$：

$$
\mathscr G=\{\prod_{i=1}^n(-\infty,a_i]:a_i \in \mathbf R^{(n)},n=1,2,\cdots,n\}
$$

甚至还可扩展至广义实数空间$ \widetilde  {\mathbf R}^{(n)}$

### 单调类

$\mathscr A$：由$\varOmega$的一些**集合**组成的非空**集合类**。
则：

- 

### 测度

## 特征函数

## n维正态分布

## 随机过程

所有样本函数的集合
时间进程中，不同时刻的随机变量的集合

### 平稳过程

1. 强平稳
$\xi(t)$的任意有限维分布函数与时间起点无关
即平稳过程的统计特性不随时间的推移而改变。

## Markov链


## 术语表

- $A$:事件集合，定义域
- $P:A \mapsto [0,1]$
- $\widetilde E$：随机试验
- $\varOmega$：$\widetilde E$的**Sample Space**，试验中所有可能结果的集合。（注：每个结果需要互斥，所有可能结果必须被穷举）。实数域的话，它的子集就是无限的。
- $\mathscr F$：由$\widetilde E$的事件（$\varOmega$的子集）组成的事件体。
- $P(A)$：集合函数


- 有限：$\cap _{i=1}^n$
- 可数/可列：
    - $\cap _{n=1}^\infty$
    - $[0,1]$不可数
- $\mathbf R^{(1)}$：一维实数空间
- $\mathbf R^{(n)}$：$n$维实数空间