---
layout:     post
title:      OpenCV Python 车道线识别
category: project
description: 识别车道线，输出黑白图片：车道线纯白（255），其他全黑。
---


## 问题

不同亮度的图片能统一

 - 检测图像白色（255）像素的比例，如果过高的话，调整相应阈值。或者，换句话是检测亮度？
	* 或者，循环内判断，跳出循环的条件就是亮度/比例
	* 对所有图片归一化？在一个范围内？
	* 看Photoshop怎么做
	* cv2的image和PIL的image区别？
		* PIL的阈值函数？

## 环境

<ul>
 <li>安装numpy、scipy、PIL（Pillow）、matplotlib</li>
 <li>安装openCV <li>		
    * 进入OpenCV的安装目录下找到：\build\python\2.7\cv2.pyd
    * 将cv2.pyd复制到Python的子目录下：\Lib\site-packages\
 <li>安装pyQt4</li>
 <li>安装pydev(Eclipse)</li>
</ul>

## 操作

参考的[解决方案][1]

### 基础操作

#### 1. 读入、输出jpg

~~~python
    for infile in  glob.glob('../photos/test/*.jpg'):
         out=processImage(infile)
         outfile=str(infile).split('\\')[1]
         outfile=outfile.split('.')[0]
         cv2.imwrite('../result/normlaneresult/'+outfile+'.jpg',out)
~~~

#### 2. **trackbar**实时调节参数

创建trackbar

~~~python
    cv2.createTrackbar('thrs1', 'fill', 2000, 10000, nothing)
~~~

从trackbar获取参数

~~~python
    while True 
        thrs1 = cv2.getTrackbarPos('thrs1', 'fill')
        cv2.imshow('fill', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
~~~

#### 3. Scharr操作

没用到，因为标识线不清晰的情况下只能突出密密麻麻的裂缝，但能有效区别开没有污损的标识线（平滑）和路面上的白色印记（粗糙）。

~~~python
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
~~~

### 平滑化处理

#### 1. 双边滤波

> 该滤波器可以在保证**边界清晰**的情况下有效的**去掉噪声**。它的构造比较复杂，即考虑了图像的**空间关系**，也考虑图像的**灰度关系**。双边滤波同时使用了空间高斯权重和灰度相似性高斯权重，确保了边界不会被模糊掉。


 `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])`
 `cv2.bilateralFilter(image, 9, 90,16)`

cv2.bilateralFilter(img,d,’p1’,’p2’)函数有四个参数需要，d是领域的直径，后面两个参数是空间高斯函数标准差和灰度值相似性高斯函数标准差。
    `cv2.bilateralFilter(img, 9, 90,16)`

#### 2. 高斯滤波（*这个好像没有太大影响。。。*）

- [ ] 必须是奇数，可以trackbar*2+1啊

~~~python
    img = cv2.GaussianBlur(image,(7,7),0)
~~~

### 对比度

OpenCV的[Histograms][2]

- [ ] 用trackbar调节参数

~~~python
    img2 = cdf[img]
    res = np.hstack((img,equ)) #stacking images side-by-side
~~~

![Image.png-7.9kB][3]

### 阈值处理

> 一般使得图像的像素值更单一、图像更简单。阈值可以分为全局性质的阈值，也可以分为局部性质的阈值，可以是单阈值的也可以是多阈值的。

#### 1.全局性的阈值

#### 2.自适应的阈值

> 通过某种算法分别为不同的区域计算不同的阈值(自适应的阈值)，然后再根据每个区域的阈值具体地去处理每个区域
![Image1.png-210kB][4]

<p>可以看到上述窗口大小使用的为11，当窗口越小的时候，得到的图像越细。想想一下，如果把窗口设置足够大以后（不能超过图像大小），那么得到的结果可能就和第二幅图像的相同了。</p>
 感觉不适合车道标线这种强调轮廓，忽视细节的对象
		`binImg = cv2.adaptiveThreshold( img , 1 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 11 , 2 ) `
该函数需要填6个参数：

<ol>
<li>原始图像</li>
<li>像素值上限</li>
<li>自适应方法Adaptive Method: </li>
  * cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值
  * cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
<li>值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV</li>
<li>Block size:规定领域大小（一个正方形的领域）</li>
<li>常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） </li>
</ol>

这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
    
~~~python
    cv2.THRESH_BINARY # 黑白二值
    binImg = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, -3)#cv2.bilateralFilter(binImg, 9, 90,16)
    #binImg = cv2.GaussianBlur(binImg, (3,3), 0)
    #ret, binImg = cv2.threshold(img, 35000, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.imshow(binImg, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
~~~

----------

## GUI部分（PyQt4）

[参考代码][5]

### 1.界面

![gui.png-6.8kB][6]

···python
    # -*- coding: utf-8 -*-
    
    import sys
    
    from PyQt4 import QtCore, QtGui 
    
    try:
        _fromUtf8 = QtCore.QString.fromUtf8
    except AttributeError:
        def _fromUtf8(s):
            return s
     
    try:
        _encoding = QtGui.QApplication.UnicodeUTF8
        def _translate(context, text, disambig):
            return QtGui.QApplication.translate(context, text, disambig, _encoding)
    except AttributeError:
        def _translate(context, text, disambig):
            return QtGui.QApplication.translate(context, text, disambig)
    
    class Ui_Dialog(object):
        def setupUi(self, Dialog):
            Dialog.setObjectName(_fromUtf8("Dialog"))
            Dialog.resize(400, 400)
            Dialog.setMinimumSize(QtCore.QSize(400, 400))
            Dialog.setMaximumSize(QtCore.QSize(400, 400))
           
            #GroupBox1
            self.groupBox = QtGui.QGroupBox(Dialog)
            self.groupBox.setGeometry(QtCore.QRect(10, 20, 381, 101))
            self.groupBox.setObjectName(_fromUtf8("groupBox"))
            self.label_2 = QtGui.QLabel(self.groupBox)
            self.label_2.setGeometry(QtCore.QRect(20, 62, 54, 12))
            self.label_2.setObjectName(_fromUtf8("label_2"))
            #Choose Input Path
            self.InputChoose = QtGui.QPushButton(self.groupBox)
            self.InputChoose.setGeometry(QtCore.QRect(320, 28, 40, 25))
            self.InputChoose.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.InputChoose.setObjectName(_fromUtf8("InputChoose"))
            self.txtPath1 = QtGui.QLineEdit(self.groupBox)
            self.txtPath1.setGeometry(QtCore.QRect(70, 30, 241, 20))
            self.txtPath1.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.txtPath1.setReadOnly(True)
            self.txtPath1.setObjectName(_fromUtf8("txtPath1"))
            #Choose Output Path
            self.OutputChoose = QtGui.QPushButton(self.groupBox)
            self.OutputChoose.setGeometry(QtCore.QRect(320, 58, 40, 25))
            self.OutputChoose.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.OutputChoose.setObjectName(_fromUtf8("OutputChoose"))
            self.txtPath2 = QtGui.QLineEdit(self.groupBox)
            self.txtPath2.setGeometry(QtCore.QRect(70, 60, 241, 20))
            self.txtPath2.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.txtPath2.setReadOnly(True)
            self.txtPath2.setObjectName(_fromUtf8("txtPath2"))
            
            #选择两种处理模式中的一个：
            self.btnLight = QtGui.QPushButton(Dialog)
            self.btnLight.setGeometry(QtCore.QRect(10, 280, 381, 51))
            self.btnLight.setObjectName(_fromUtf8("btnLight"))
            
            self.btnDark = QtGui.QPushButton(Dialog)
            self.btnDark.setGeometry(QtCore.QRect(10, 340, 381, 51))
            self.btnDark.setObjectName(_fromUtf8("btnDark"))
        
            self.retranslateUi(Dialog)
            QtCore.QMetaObject.connectSlotsByName(Dialog)
        
        def retranslateUi(self, Dialog):
            Dialog.setWindowTitle(_translate("Dialog", "车道标线识别程序 by Rowling", None))
            self.btnLight.setText(_translate("Dialog", "光亮模式", None))
            self.btnDark.setText(_translate("Dialog", "昏暗模式", None))
            self.groupBox.setTitle(_translate("Dialog", "请选择需要处理的图片文件夹以及输出图片的存放位置", None))
            self.label_2.setText(_translate("Dialog", "输出", None))
            self.InputChoose.setText(_translate("Dialog", "...", None))
            self.OutputChoose.setText(_translate("Dialog", "...", None))
        
       
           
    if __name__ == '__main__':
        app = QtGui.QApplication(sys.argv)
        Dialog = QtGui.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        sys.exit(app.exec_())
        #应用程序的主事件循环，事件处理从这里开始
···

- exec是Python的关键字，因此，用 exec_() 来取代它。

#### 2. 自定义槽

···python
        # -*- coding: utf-8 -*-
 
import sys

from PyQt4 import QtGui
from PyQt4 import QtCore
 
import Fill
from GUI import Ui_Dialog
 
class ImageProcess(QtGui.QDialog, Ui_Dialog):
    def __init__(self, parent = None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.exclude_list = []
    
    @QtCore.pyqtSignature("")
    def on_InputChoose_clicked(self):
        self.indir = QtGui.QFileDialog.getExistingDirectory(self,  u"选择目录","../photos/light_lane").replace("\\","/")
        self.txtPath1.setText(self.indir)
    
    @QtCore.pyqtSignature("")
    def on_OutputChoose_clicked(self):
        self.outdir = QtGui.QFileDialog.getExistingDirectory(self,  u"选择目录","../result/laneresult").replace("\\","/")
        self.txtPath2.setText(self.outdir)
 
   
    @QtCore.pyqtSignature("")
    def on_btnLight_clicked(self):
        Fill.processLightLane(str(self.indir),str(self.outdir))
        QtGui.QMessageBox.warning(self, '完成', '处理完毕，请到输出目录下查看')
    
     
    @QtCore.pyqtSignature("")
    def on_btnDark_clicked(self):
        Fill.processDarkLane(str(self.indir),str(self.outdir))
        QtGui.QMessageBox.warning(self, '完成', '处理完毕，请到输出目录下查看')
        
    def get_formated_path1(self):
        return self.format_path(self.txtPath1.text())
        
    def get_formated_path2(self):
        return self.format_path(self.txtPath2.text())
        
    def format_path(self, path
        return str(path).rstrip('\\') + '\\'
 
 
if __name__ == "__main__":
    reload(sys)
    app = QtGui.QApplication(sys.argv)
    dlg = ImageProcess()
    dlg.show()
    sys.exit(app.exec_())
 
···

### 3. 核心函数文件


## 打包成EXE

可以看这个[视频][7]，有好几种选择：

 - py2exe 
 - cx_Freeze
 - pyinstaller

 这里来安装pyinstaller,按Documentation的要求，首先就是要安装pywin32,选择合适的版本即可， 比如我的Python是2.7，系统是64位。
安装完后，直接用powershell(管理员模式下打开)，运行`pip install PyInstaller`。
之后切到要打包的python程序的文件夹运行`pyinstaller XXX.py`即可，目标exe保存在dist目录下。
补充：

>  - -c, --console, --nowindowed  	Open a console window for standard i/o (default)
>  - -w, --windowed, --noconsole  	Windows and Mac OS X: **do not** provide a **console window** for standard i/o. On Mac OS X this also triggers
> building an OS X .app bundle.This option is ignored in *NIX systems.
>  - -F, --onefile	Create a one-file bundled executable.
>  - -D, --onedir	Create a one-folder bundle containing an executable (default)

所以最后的执行语句是：

    pyinstaller -F -D --noconsole -n Lane G:\C\Lane\Lane\file_ui.py

没执行成功呃。。。咋回事儿?

----------

## 花絮

这次用的是全局性阈值，自适应阈值能平衡细节，但对有裂缝的车道来说并不适用。因为最后的结果强调标志线与路面的黑白对比，所以只是通过一个全局的阈值“筛出”白色标志线。不过这时遇到一个问题：对于光照条件不同的路面照片，阈值是不同的,甚至在一种光照条件（暗光），路面照片也不一样，这个时候开始考虑归一化和增强对比度的问题。
真傻还是假傻。。。。在powershell里执行命令，默认都是在system32文件夹里啊。。。。得切到工程文件夹里才好啊啊啊啊

-  **ToDo List**
    - [ ] 其中一个环节就是通过track bar来找到合适的滤波参数（cdf的对比度没参数……）
    - [ ] 下一步的思路是：模糊化增强（双边滤波）-》增强对比度-》阈值-》输出
    - [x] 用cdf增强对比度后发现路面上稍亮一些的印记也被强化了。

- 参考

清华大学出版社的《图像处理与计算机视觉算法及应用(第0版) 》TP391.41/P121.

图书馆还有一本但就是找不到。。。


。


  [1]: http://stackoverflow.com/questions/26145806/filling-contours-in-opencv
  [2]: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
  [3]: http://static.zybuluo.com/sixijinling/b5ze3we16g5z2t61qf6k5uwd/Image.png
  [4]: http://static.zybuluo.com/sixijinling/8dy1l0qe9n3y7tjgukglu51t/Image1.png
  [5]: http://www.pythoner.com/176.html
  [6]: http://static.zybuluo.com/sixijinling/fq1y0onoen7c8hr6x2n0yzed/gui.png
  [7]: https://www.youtube.com/watch?v=11Q2QADsAEE
