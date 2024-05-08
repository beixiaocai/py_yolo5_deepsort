# py_yolo5_deepsort
* github代码地址：https://github.com/any12345com/py_yolo5_deepsort

### 环境依赖

| 程序         | 版本               |
| ---------- |------------------|
| python     | 3.8+            |
| 依赖库      | requirements.txt |
### 介绍
* 目标检测+追踪，python+pytorch推理yolo5，python+pytorch推理deepsort

### 启动
~~~
python main.py

~~~


### Windows系统安装 pytorch-cpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### Windows系统安装 pytorch-gpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
* 注意：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择


### 常见问题总结

~~~
//问题 AttributeError: module ‘numpy‘ has no attribute ‘float‘

//分析问题：出现这个问题的原因是，从numpy1.24起删除了numpy.bool、numpy.int、numpy.float、numpy.complex、numpy.object、numpy.str、numpy.long、numpy.unicode类型的支持。解决上诉问题主要有两种方法：

//解决问题：参考https://blog.csdn.net/qq_51511878/article/details/129811061

方法一：修改numpy版本
安装numpy1.24之前的版本
pip uninstall numpy
pip install numpy==1.23.5

方法二：修改代码
可以用python内置类型或者np.ndarray类型替换：

np.float替换为float或者np.float64/np.float32

~~~


