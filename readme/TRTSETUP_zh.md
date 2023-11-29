# TensorRT 安装教程

TensorRT 是由 NVIDIA 开发的深度学习推理引擎，旨在优化和加速深度学习模型的推理过程。它是为了在 NVIDIA GPU 上部署和高效执行深度学习模型而设计的。利用 Pytorch、TensorFlow 等框架训练好的模型，可以转化为 TensorRT 的格式，然后利用 TensorRT 推理引擎进行推理，从而提升该模型在 GPU 上的运行速度。对于部署于 GPU 的深度学习模型，TensorRT 是个非常理想的推理框架。

本教程提供了在 Windows 系统中 TensorRT 的安装及配置方法。本机的显卡型号为 `NVIDIA GeForce RTX 4080 Laptop GPU`，使用的环境为：
* CUDA 11.7
* cuDNN 8.8
* TensorRT 8.5
* Python 3.8.13

## 1 环境准备
TensorRT 与 CUDA 及 cuDNN 的版本是对应的，因此需要先确定已安装好 CUDA 与 cuDNN，并确认其版本。 <br>
CUDA 下载地址：https://developer.nvidia.com/cuda-downloads <br>
cuDNN 下载地址：https://developer.nvidia.com/rdp/cudnn-download <br>

### 1.1 检查 CUDA
进入命令行，输入 `nvcc -V`，若看到如下信息，则说明 CUDA 已安装。
```
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```
进一步，进入 CUDA 的安装路径，默认情况下是 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`，进入其下的 `v11.7\extras\demo_suite`，其中 v11.7 为下载的 CUDA 版本号。在该路径下打开命令行，运行 `.\deviceQuery.exe` 及 `.\bandwidthTest.exe`，若两项测试均通过，说明 CUDA 能正常工作。

P.S. 若 CUDA 未安装好，根据自己显卡的型号到官网下载合适版本的 CUDA 即可。

### 1.2 检查 cuDNN
cuDNN 的安装是将一系列文件导入到 CUDA 的安装目录下：
* 从官网下载好与 CUDA 对应版本的 cuDNN 安装包，官网要求注册会员并登录后方可下载。解压至本地；
* 将 cuDNN 中 `bin` 目录下的文件移动到 CUDA 的 `bin` 目录（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin`）；
* 将 cuDNN 中 `include` 目录下的文件移动到 CUDA 的 `include` 目录；
* 将 cuDNN 中 `lib\x64` 目录下的文件移动到 CUDA 的 `lib\x64` 目录。

## 2 安装 TensorRT
### 2.1 下载
TensorRT 可以通过官网 https://developer.nvidia.com/tensorrt 下载，官网要求注册会员并登录后方可下载。Windows 系统只需下载好压缩包并解压到本地即可，注意选择与 CUDA 版本对应的 TensorRT 版本。

### 2.2 配置
#### 2.2.1. 文件配置
TensorRT 的配置与 cuDNN 的配置类似：
* 将 TensorRT 安装包中 `include` 目录下的文件复制到 CUDA 的 `include` 目录下；
* 将 TensorRT 安装包中 `lib` 目录下的所有 `lib` 文件复制到 CUDA 的 `lib\x64` 目录下；
* 将 TensorRT 安装包中 `lib` 目录下的所有 `dll` 文件复制到 CUDA 的 `bin` 目录下；
* 将 TensorRT 的 `bin` 路径添加到环境变量；
* 将 CUDA 的 `include`、`lib` 和 `bin` 路径添加到环境变量。

#### 2.2.2 安装 tensorrt
* 进入 `TensorRT-8.5.1.7\python` 目录，该目录下有 tensorrt 的针对不同 python 版本的 `whl` 文件。我们虚拟环境中的 python 版本为 3.8，应该安装 `cp38` 对应的 `whl` 文件。
  在该路径下打开终端命令行，进入虚拟环境后，运行：
  ```
  pip install tensorrt -8.5.1.7 - cp38-none -win_amd 64. whl
  ```
* 进入 `TensorRT-8.5.1.7\graphsurgeon` 目录，在该路径下打开终端命令行，进入虚拟环境后，运行：
  ```
  pip install graphsurgeon -0.4.6 - py2.py3-none -any.whl
  ```
* 进入 `TensorRT-8.5.1.7\onnx_graphsurgeon` 目录，在该路径下打开终端命令行，进入虚拟环境后，运行：
  ```
  pip install onnx_ graphsurgeon -0.3.12 - py2.py3-none -any.whl
  ```
安装完成后，进入 python 环境，打印版本号等信息，若不报错则说明安装成功。
```
import tensorrt as trt
print(trt .__ version __)
assert trt. Builder (trt. Logger ())
```

#### 2.2.3 安装 pycuda
* 在 PyCUDA 下载网址 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda 上根据 CUDA 和 python 版本下载好合适的 PyCUDA `whl` 文件。
* 进入 `whl` 文件所在目录，在该路径下打开终端命令行，进入虚拟环境后，运行：
```
pip install pycuda ‑2022.1+ cuda 116‑ cp 38‑ cp 38‑ win_amd 64. whl
```

以上一系列的配置完成后，在使用 TensorRT 的编译工具对 ONNX 模型进行转换时，仍可能会报错：`Could not locate zlibwapi.dll. Please make sure it is in your library path!` <br>
解决方案是：
* 首先下载 `zlib` 文件，解压后进入 `dll_x64` 文件夹；
* 将 `zlibwapi.lib` 文件放到 `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v11.7\lib\x64` 下；
* 将 `zlibwapi.dll` 文件放到 `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v11.7\bin` 下。


## 参考文章
[1] [Windows 安装 CUDA / cuDNN](https://zhuanlan.zhihu.com/p/99880204?from_voters_page=true) <br>
[2] [Windows 系统下如何确认 CUDA 和 cuDNN 都安装成功了](https://blog.csdn.net/qq_35768355/article/details/132985948) <br>
[3] [TensorRT 安装](https://blog.csdn.net/weixin_51691064/article/details/130403978) <br>
[4] [TensorRT 安装 zlibwapi.dll](https://blog.csdn.net/weixin_42166222/article/details/130625663) <br>
[5] [TensorRT 安装记录](https://blog.csdn.net/qq_37541097/article/details/114847600) <br>
[6] [PyCUDA 安装与使用](https://blog.csdn.net/qq_41910905/article/details/109650182)
