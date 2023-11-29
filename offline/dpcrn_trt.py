""" 
1. onnx to engine: use `trtexec.exe`.
2. tensorrt and torch cannot be imported simultaneously!
"""
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # must import
from collections import namedtuple

def onnx2engine(trtexec_path, onnx_path, save_path):
    os.system(f"{trtexec_path} --onnx={onnx_path} --saveEngine={save_path}")


Bindings = namedtuple("Bindings", ("name", "shape", "host", "device", "size"))

class TRTModel:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path, dtype=np.float32):
        """
        Args:
            engine_path: The path to the serialized engine to load from disk.
            dtype: The datatype used in inference.
        """
        # init arguments
        self.engine_path = engine_path
        self.dtype = dtype

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine


    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            # print(i, name, ':', shape, size)

            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(Bindings(name, shape, host_mem, device_mem, host_mem.nbytes))
            else:
                outputs.append(Bindings(name, shape, host_mem, device_mem, host_mem.nbytes))

        return inputs, outputs, bindings, stream


    def __call__(self, x: np.ndarray):
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())

        # Transfer the noisy data from CPU to CUDA.
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        # Execute inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer the enhanced data from CPU to CUDA.
        cuda.memcpy_dtoh_async(self.outputs[-1].host, self.outputs[-1].device, self.stream)

        self.stream.synchronize()

        return self.outputs[-1].host.reshape(self.outputs[-1].shape)
    

if __name__ == "__main__":
    import time

    trtexec_path = r'.\models\trtexec.exe'
    onnx_path = r'.\models\dpcrn.onnx'
    save_path = r'.\models\dpcrn.engine'

    ## Convert to engine
    onnx2engine(trtexec_path, onnx_path, save_path)

    ## Load engine model
    model = TRTModel(save_path)

    ## Execute inference
    time_len = 9  # set the length to 9 s
    frame_num = time_len * 16000 // 256 + 1  # compute frame numbers, fs=16000, hop_size=256
    x = np.random.randn(1, 257, frame_num, 2)
    y = model(x)

    ## Test inference speed
    times = np.zeros([1000])
    for i in range(len(times)):
        tic = time.perf_counter()
        outputs = model(x)
        toc = time.perf_counter()
        times[i] = 1000*((toc-tic) / time_len)

    print(">>> Average Inference Time (ms): ", times.mean())
    print(">>> Maximum Inference Time (ms): ", times.max())
    print(">>> Minimum Inference Time (ms): ", times.min())




    

