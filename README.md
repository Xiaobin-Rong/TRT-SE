# Deploying Deep Learning Speech Enhancement Models with TensorRT
The Chinese version of the document can be found in: [使用 TensorRT 部署深度学习语音增强模型](./readme/README_zh.md)

To install TensorRT, refer to: [TensorRT Installation Tutorial](./readme/TRTSETUP.md)

The deployment of speech enhancement models can be categorized into two types: **offline inference** and **online inference**. Offline inference involves performing model inference on pre-prepared data, usually in the form of a batch of samples or longer audio signals. Offline inference does not have real-time requirements and can utilize efficient inference methods and resource allocation strategies.

On the other hand, online inference involves performing model inference on real-time generated speech data, such as continuously captured audio signals from a microphone. Online inference requires low latency and high throughput to meet real-time requirements.

## Deployment of Offline Models
### 1. Conversion to ONNX Model
For offline models, exporting to ONNX is straightforward. The only thing to consider is setting the time dimension of the input shape. Although `torch.onnx.export` supports dynamic dimensions, considering the limited need in practical applications, we choose to fix the time dimension to 563, corresponding to 9 seconds of audio data. During offline processing, if the audio is less than 9 seconds, it is padded with zeros. If the audio is longer than 9 seconds, it is processed in batches of 9 seconds each.

`offline\dpcrn_onnx.py` provides the export and inference of the ONNX model, and evaluates the inference speed on ONNXRuntime.

### 2. Conversion to Engine Model
We use the conversion tool `trtexec.exe` provided by TensorRT to convert the model from ONNX to the Engine format supported by TensorRT. This tool is located in the `bin` directory of the TensorRT installation package, and the usage is as follows:
```
trtexec.exe --onnx=[onnx_path] --saveEngine=[save_path]
```

`offline\dpcrn_trt.py` provides the export and inference of the Engine model, and evaluates the inference speed. The results are shown in the following table.

| **Model Format** | **Inference Framework** | **Inference Platform** | **Average Inference Speed (ms)** | **Maximum Inference Speed (ms)** | **Minimum Inference Speed (ms)** |
|:----------------:|:----------------------:|:---------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| ONNX | ONNXRuntime | CPU | 8.6 | 21.0 | 7.6|
| Engine |TensorRT| CUDA | 2.2 | 5.1 | 1.9 |

The evaluation of inference speed is repeated 1000 times. Here, the inference speed is defined as the processing time per audio duration. It can be observed that the inference speed of the TensorRT framework is nearly 4 times faster than ONNXRuntime.

## Deployment of Online Models
In speech enhancement, online inference has broader application scenarios and higher real-time requirements for the model. Consequently, the deployment of online inference is more complex. Here, we adopt the method of **streaming inference** to perform frame-by-frame inference on real-time data streams. When implementing streaming inference, appropriate data buffering mechanisms, data stream management, and pipeline design for model inference are required to ensure data continuity and stable inference.

### 1. Conversion to Streaming Model
RNN naturally adapts to streaming inference without additional conversion. In contrast, the convolutional layers are the main part of the neural network that requires conversion for streaming. In the `online\modules` directory, we define two types of operators for streaming convolution and streaming transposed convolution in `convolution.py`. We also provide a method in `convert.py` to copy the original model parameter dictionary for the conversion of streaming models.

`online\dpcrn_stream.py` provides the conversion and inference process for streaming models. Note that for streaming models, the time dimension of the input tensor is always set to 1.

### 2. Conversion to ONNX Model
For streaming models, there is no need to consider the time dimension when converting to ONNX. However, it is recommended to specify all input tensors in the `forward` function instead of using a list as done in `online\dpcrn_stream.py`.

`online\dpcrn_stream_onnx.py` provides the conversion and inference process for streaming ONNX models and evaluates the inference speed on ONNXRuntime.

### 3. Conversion to Engine Model
Similarly, we use the conversion tool `trtexec.exe` provided by TensorRT for model conversion.

`online\dpcrn_stream_trt.py` provides the export and inference of the streaming Engine model, and evaluates the inference speed. The results are shown in the following table.

| **Model Format** | **Inference Framework** | **Inference Platform** | **Average Inference Speed (ms)** | **Maximum Inference Speed (ms)** | **Minimum Inference Speed (ms)** |
|:----------------:|:----------------------:|:---------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| ONNX | ONNXRuntime | CPU |1.0 | 3.1 | 0.9 |
| Engine |TensorRT| CUDA |2.2 | 4.7 | 1.8 |

The evaluation of inference speed is repeated 1000 times. Here, the inference speed is defined as the processing time per frame. As we can see, using TensorRT for inference is slower than using ONNXRuntime. This is because, for high-throughput streaming models, the data transfer between CUDA and CPU during TensorRT inference takes a certain amount of time. Only when the model's inference speed on the CPU becomes the bottleneck, using TensorRT for CUDA inference will have a positive effect.

