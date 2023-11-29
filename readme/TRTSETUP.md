# TensorRT Installation Tutorial
The Chinese version of the document can be found in: [TensorRT 安装教程](./TRTSETUP_zh.md)

TensorRT is a deep learning inference engine developed by NVIDIA, designed to optimize and accelerate the inference process of deep learning models. It is specifically designed for deploying and efficiently executing deep learning models on NVIDIA GPUs. Models trained using frameworks such as PyTorch and TensorFlow can be converted to the TensorRT format and then utilized with the TensorRT inference engine to improve the model's runtime speed on GPUs. TensorRT is an ideal framework for deploying deep learning models on GPUs.

This tutorial provides the installation and configuration methods for TensorRT on the Windows operating system. I use `NVIDIA GeForce RTX 4080 Laptop GPU`, and the corresponding software environment consists of the following components:
* CUDA 11.7
* cuDNN 8.8
* TensorRT 8.5
* Python 3.8.13

## 1. Environment Setup
TensorRT is version-specific to CUDA and cuDNN, so it is necessary to ensure that CUDA and cuDNN are already installed and their versions are compatible. <br>
CUDA download link: https://developer.nvidia.com/cuda-downloads <br>
cuDNN download link: https://developer.nvidia.com/rdp/cudnn-download <br>

### 1.1 Check CUDA Installation
Open the command prompt and enter `nvcc -V`. If you see the following information, it means CUDA is installed:
```
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```
Furthermore, navigate to the installation path of CUDA, which is by default `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`. Within this path, go to `v11.7\extras\demo_suite`, where `v11.7` represents the version of CUDA downloaded. In this directory, open the command prompt and run `.\deviceQuery.exe` and `.\bandwidthTest.exe`. If both tests pass, it indicates that CUDA is functioning correctly.

Note: If CUDA is not installed, download the appropriate version for your GPU model from the official NVIDIA website.

### 1.2 Check cuDNN Installation
Installing cuDNN involves copying a series of files to the CUDA installation directory:
* Download the cuDNN installation package corresponding to your CUDA version from the official website. Registration and login are required on the website to access the downloads. Extract the files locally.
* Move the files in the `bin` directory of cuDNN to the `bin` directory of CUDA (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin`).
* Move the files in the `include` directory of cuDNN to the `include` directory of CUDA.
* Move the files in the `lib\x64` directory of cuDNN to the `lib\x64` directory of CUDA.

## 2. Install TensorRT
### 2.1 Download
TensorRT can be downloaded from the official website https://developer.nvidia.com/tensorrt. Registration and login are required on the website to access the downloads. For Windows systems, simply download the compressed package and extract it to the local machine, making sure to select the TensorRT version that corresponds to your CUDA version.

### 2.2 Configuration
#### 2.2.1 File Configuration
   The configuration process for TensorRT is similar to that of cuDNN:
   * Copy the files in the `include` directory of the TensorRT installation package to the `include` directory of CUDA.
   * Copy all the `lib` files in the `lib` directory of the TensorRT installation package to the `lib\x64` directory of CUDA.
   * Copy all the `dll` files in the `lib` directory of the TensorRT installation package to the `bin` directory of CUDA.
   * Add the path to the `bin` directory of TensorRT to the environment variables.
   * Add the paths to the `include`, `lib`, and `bin` directories of CUDA to the environment variables.

#### 2.2.2 Install tensorrt
   * Navigate to the `TensorRT-8.5.1.7\python` directory, which contains the `whl` files for different versions of TensorRT compatible with different Python versions. Since the Python version in the virtual environment is 3.8, install the `whl` file corresponding to `cp38`. In the terminal command line at this path, within the virtual environment, run:
     ```
     pip install tensorrt-8.5.1.7-cp38-none-win_amd64.whl
     ```
   * Navigate to the `TensorRT-8.5.1.7\graphsurgeon` directory. In the terminal command line at this path, within the virtual environment, run:
     ```
     pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
     ```
   * Navigate to the `TensorRT-8.5.1.7\onnx_graphsurgeon` directory，In the terminal command line at this path, within the virtual environment, run:
      ```
      pip install onnx_ graphsurgeon -0.3.12 - py2.py3-none -any.whl
      ```
    After the installation is complete, enter the Python environment and print the version information. If no errors are reported, it indicates a successful installation.
    ```
    import tensorrt as trt
    print(trt .__ version __)
    assert trt. Builder (trt. Logger ())
    ```

#### 2.2.3 Install PyCUDA
   * Go to the PyCUDA download URL: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda and download the appropriate PyCUDA `.whl` file based on your CUDA and Python versions.
   * Navigate to the directory where the `.whl` file is located. Open a terminal or command prompt at that path and activate your virtual environment.
   * Run the following command:
     ```
     pip install pycuda‑2022.1+cuda116‑cp38‑cp38‑win_amd64.whl
     ```

After completing the above configurations, you may still encounter an error when using the TensorRT compilation tool to convert ONNX models: `Could not locate zlibwapi.dll. Please make sure it is in your library path!` 

To resolve this issue:
* First, download the `zlib` file and extract its contents.
* Navigate to the `dll_x64` folder.
* Copy the `zlibwapi.lib` file and paste it into `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v11.7\lib\x64`.
* Copy the `zlibwapi.dll` file and paste it into `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v11.7\bin`.


## Reference
[1] [Windows 安装 CUDA / cuDNN](https://zhuanlan.zhihu.com/p/99880204?from_voters_page=true) <br>
[2] [Windows 系统下如何确认 CUDA 和 cuDNN 都安装成功了](https://blog.csdn.net/qq_35768355/article/details/132985948) <br>
[3] [TensorRT 安装](https://blog.csdn.net/weixin_51691064/article/details/130403978) <br>
[4] [TensorRT 安装 zlibwapi.dll](https://blog.csdn.net/weixin_42166222/article/details/130625663) <br>
[5] [TensorRT 安装记录](https://blog.csdn.net/qq_37541097/article/details/114847600) <br>
[6] [PyCUDA 安装与使用](https://blog.csdn.net/qq_41910905/article/details/109650182)
