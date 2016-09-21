---
layout: post
title: 非线性最优化——说人话
description: 依然感谢头脑不清不出坚持上课的我，说得像身残志坚一样
category: blog
---

## Reference Books

- 《最优化理论与方法》陈宝林
- 《非线性优化计算方法》袁亚湘
-  《Numerical Optimization》Jorge Nocedal

梯度（gradient）：导数的高维扩展
最速下降法：负梯度方向，“速”非快

1. local minimun $\Longrightarrow \nabla f(x^*)=0,\nabla ^2 f(x^*) \geq 0$ 
2. $\nabla f(x^*)=0 \& \nabla ^2 f(x^*) >0 \Longrightarrow  $local minimum ,eg. $y=x^3$

## KKT条件

In mathematical optimization, the **Karush–Kuhn–Tucker** (KKT) conditions (also known as the **Kuhn–Tucker conditions**) are first order necessary conditions for a solution in nonlinear programming to be optimal, provided that some regularity conditions are satisfied. Allowing inequality constraints, the KKT approach to nonlinear programming generalizes the method of Lagrange multipliers, which allows only equality constraints. The system of equations corresponding to the KKT conditions is usually not solved directly, except in the few special cases where a closed-form solution can be derived analytically. In general, many optimization algorithms can be interpreted as methods for numerically solving the KKT system of equations.[1]

The KKT conditions were originally named after Harold W. Kuhn, and Albert W. Tucker, who first published the conditions in 1951.[2] Later scholars discovered that the necessary conditions for this problem had been stated by William Karush in his master's thesis in 1939.

## 无约束

### 0.618法

用0.618法寻找最佳点时，虽然不能保证在有限次内准确找出最佳点，但随着试验次数的增加，最佳点被限定在越来越小的范围内，即存优范围会越来越小。用存优范围与原始范围的比值来衡量一种试验方法的效率，这个比值叫精度。用0.618法确定试点时，每一次实验都把存优范围缩小为原来的0.618.因此，n次试验后的精度为：

$$
\delta _n=0.618^n
$$

### 1. 梯度法（最速下降法）

### 2. 共轭梯度法

### 3. Newton法

#### 一维

![牛顿法][1]

$$
x_{k+1}=x_k-\frac {f(x_k)}{f'(x_k)}
$$

#### 推广至高维

Hessian矩阵$\mathbf H=\nabla^2 f(\mathbf x)$
$$
\mathbf x_{k+1}=\mathbf x_k-\frac {\nabla f(\mathbf x_k)}{\nabla^2f(\mathbf x_k)}
$$

### 线搜索

### bb法

## 有约束



  [1]: https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/NewtonIteration_Ani.gif/300px-NewtonIteration_Ani.gif