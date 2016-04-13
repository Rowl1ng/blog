---
layout: post
title: 微小区公众号开发
category: project
description: 别人做的啦，但是需求改了就要改代码。。。。。今天开始学PHP
---


在Ubuntu上搭建环境：

    sudo apt-get install tasksel
    sudo tasksel
选择lamp服务器（空格输入 * 代表选中）,之后就会自动安装相关的套件。安装mysql时需要输入密码。
安装完成后就可以在浏览器中输入127.0.0.1来进行测试了。


![2016-03-02 16-18-21屏幕截图.png-236.8kB][1]

    mysql -uroot -p
    mysql> create database village;
    mysql> use village;
    mysql> set names utf8;
    mysql> show tables; 
打算配一下PHPadmin,方便看数据库。先来安装PHP：

    sudo apt-get install php5 libapache2-mod-php5
    sudo service apache2 restart //重启一下服务
下面编辑一个php文件验证一下：

    sudo vim /var/www/html/phpinfo.php
内容：
```php
    <?php  
       phpinfo();  
    ?> 
```
在浏览器中访问`http://localhost/phpinfo.php`
![2016-03-02 21-21-28屏幕截图.png-219.9kB][2]

下面来装mysql的图形界面phpadmin：

    sudo apt-get install phpmyadmin
有提示选择web服务器，选择apache。
phpmyadmin安装完后，并不在apache默认路径下，需要建立一个连接：
    
    sudo ln -s /usr/share/phpmyadmin /var/www/html
重启apache服务器，浏览器打开：`http://localhost/phpmyadmin`
![2016-03-02 21-28-26屏幕截图.png-146.4kB][3]


    
## PHPStorm

### Checking out Files from a Repository
#### SVN

在终端查看svn安装位置：
    >which svn
    >/usr/bin/svn
#### cgi

    sudo apt-get install php5-cgi


----------
## 进入正题
**ThinkPHP 3.2**版本采用模块化的设计架构，下面是一个应用目录下面的模块目录结构，每个模块可以方便的卸载和部署，并且支持公共模块。

    Application      默认应用目录（可以设置）
        ├─Common         公共模块（不能直接访问）
        ├─Home           前台模块
        ├─Admin          后台模块
        ├─...            其他更多模块
        ├─Runtime        默认运行时目录（可以设置）
        
每个模块是相对独立的，其目录结构如下：

    ├─Module         模块目录
    │  ├─Conf        配置文件目录
    │  ├─Common      公共函数目录
    │  ├─Controller  控制器目录
    │  ├─Model       模型目录
    │  ├─Logic       逻辑目录（可选）
    │  ├─Service     Service目录（可选）
    │  ... 更多分层目录可选
    │  └─View        视图目录
### 1. Common公共模块
Common模块是一个特殊的模块，是应用的公共模块，访问所有的模块之前都会首先加载公共模块下面的配置文件（Conf/config.php）和公共函数文件（Common/function.php）。但Common模块本身不能通过URL直接访问，公共模块的其他文件则可以被其他模块继承或者调用。

  [1]: http://static.zybuluo.com/sixijinling/hh6hogyza7mluss3xgpd5ga4/2016-03-02%2016-18-21%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [2]: http://static.zybuluo.com/sixijinling/qci9tn5q8ulq2o3de1sxp0hk/2016-03-02%2021-21-28%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [3]: http://static.zybuluo.com/sixijinling/g0kg30ewh5zzjangpm9b64ck/2016-03-02%2021-28-26%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
