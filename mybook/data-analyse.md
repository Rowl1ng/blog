# 必先利其器

## Pycharm远程连接服务器调试

使用

你应该对Python的科学计算包和深度学习包有一定了解，这些包包含但不限于numpy, scipy, scikit-learn, pandas...

## Jupyter

远程访问服务器上的jupyter需设置密码：

```
pip install jupyter --user
jupyter notebook --generate-config

In [1]: from notebook.auth import passwd
In [2]: passwd() 
Enter password: 
Verify password: 

sha1:4ebe51569f06:e2030b8480f6b1f07d920fdbf9851101b5ce45da

vi ~/.jupyter/jupyter_notebook_config.py
进行如下修改：
c.NotebookApp.ip='*' # 就是设置所有ip皆可访问
c.NotebookApp.password = u'sha:73...刚才复制的那个密文'
c.NotebookApp.open_browser = False # 禁止自动打开浏览器
c.NotebookApp.port =8888 #随便指定一个端口
```

运行：

```
jupyter notebook --ip=192.168.8.150 --port=xxxx
```

[安装多个版本python的kernel](https://www.jianshu.com/p/e140c5c97938)

编码问题：

```
# 有中文
XXX.decode('utf-8')
```

## Reference

* [Pandas & Seaborn - A guide to handle & visualize data in Python\][2]

[2]: [https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/](https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/)

