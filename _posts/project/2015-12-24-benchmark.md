---
layout:     post
title:      word2vec Benchmark
category: project
description: 暑假9月到12月实习就在做这个，现在适当总结一下吧，主要也就是technical report的内容。总觉得写的不好，一边整理一边改改。
---


---

##Introduction

The success of machine learning methods in NLP tasks depends much on word representation, since different representations may encode different explanatory factors of variation behind the word. With the rapid development of deep learning techniques,researchers have started to train complex and deep models on large amounts of text corpus, to learn distributed representations of words(also known as word embeddings) in the form of continuous vectors.
While conventional NLP techniques usually represent words as indices in a vocabulary causing no notion of relationship between words, word embeddings learned by deep learing approaches aim at explicitly encoding many semantic relationships as well as linguistic regularities and patterns into the new word embedding space.

 不同于传统one-hot，distributed词向量包含了词与词之间的关联

In this paper, we introduce a benchmark collection\cite{how}, which is built from several different data source, to measure quality of word embeddings from different aspects.
复现了若干个task，来衡量词向量的优劣。

##Models

### 1. NNLM

这里看的是Bengio的[论文][1]

Bengio et al. \cite{nnlm} first proposed a Neural Network Language Model (NNLM) that simultaneously learns a word embedding and a language model.The language model utilizes several previous words to predict the distribution of the next word.For each sample in the corpus ,we maximize the log-likelihood of the probability of the last word given the previous words.This model uses a concatenation of the previous words' embeddings as the input.The model structure is a feed-forward neural network with one hidden layer.

### 2. LBL

The Log-Bilinear Language Model(LBL) proposed by Mnih and Hinton combines Bengio's Hierachical NNLM and Log Bi-Linear.It uses a log-bilinear energy function that is almost equal to that of the NNLM and removes the non-linear activation function tanh. 

A previous study \cite{lbl} proposed a widely used model architecture for estimating neural network language model.

### NER



  [1]: http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
