""" 
1. onnx to engine: use `trtexec.exe`.
2. tensorrt and torch cannot be imported simultaneously!
3. During streaming inference of the engine model, if we set outputs=[] and then append frame by frame,
it will result in each element in outputs being the result of the last frame (reason unknown).
The solution is to set outputs=np.zeros((1, 2, T, F)) and then assign values frame by frame.
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
            ## 0 mix : (1, 257, 1, 2) 514
            ## 1 en_cache1 : (1, 2, 1, 257) 514
            ## 2 en_cache2 : (1, 32, 1, 129) 4128
            ## 3 en_cache3 : (1, 32, 1, 65) 2080
            ## 4 en_cache4 : (1, 32, 1, 33) 1056
            ## 5 en_cache5 : (1, 64, 1, 33) 2112
            ## 6 rnn_cache1 : (1, 33, 128) 4224
            ## 7 rnn_cache2 : (1, 33, 128) 4224
            ## 8 rnn_cache3 : (1, 33, 128) 4224
            ## 9 rnn_cache4 : (1, 33, 128) 4224
            ## 10 de_cache1 : (1, 256, 1, 33) 8448
            ## 11 de_cache2 : (1, 128, 1, 33) 4224
            ## 12 de_cache3 : (1, 64, 1, 33) 2112
            ## 13 de_cache4 : (1, 64, 1, 65) 4160
            ## 14 de_cache5 : (1, 64, 1, 129) 8256
            ## 15 en_cache1_out : (1, 2, 1, 257) 514
            ## 16 en_cache2_out : (1, 32, 1, 129) 4128
            ## 17 en_cache3_out : (1, 32, 1, 65) 2080
            ## 18 en_cache4_out : (1, 32, 1, 33) 1056
            ## 19 en_cache5_out : (1, 64, 1, 33) 2112
            ## 20 rnn_cache1_out : (1, 33, 128) 4224
            ## 21 rnn_cache2_out : (1, 33, 128) 4224
            ## 22 rnn_cache3_out : (1, 33, 128) 4224
            ## 23 rnn_cache4_out : (1, 33, 128) 4224
            ## 24 de_cache1_out : (1, 256, 1, 33) 8448
            ## 25 de_cache2_out : (1, 128, 1, 33) 4224
            ## 10 de_cache1 : (1, 256, 1, 33) 8448
            ## 11 de_cache2 : (1, 128, 1, 33) 4224
            ## 12 de_cache3 : (1, 64, 1, 33) 2112
            ## 13 de_cache4 : (1, 64, 1, 65) 4160
            ## 14 de_cache5 : (1, 64, 1, 129) 8256
            ## 15 en_cache1_out : (1, 2, 1, 257) 514
            ## 16 en_cache2_out : (1, 32, 1, 129) 4128
            ## 17 en_cache3_out : (1, 32, 1, 65) 2080
            ## 18 en_cache4_out : (1, 32, 1, 33) 1056
            ## 19 en_cache5_out : (1, 64, 1, 33) 2112
            ## 20 rnn_cache1_out : (1, 33, 128) 4224
            ## 21 rnn_cache2_out : (1, 33, 128) 4224
            ## 22 rnn_cache3_out : (1, 33, 128) 4224
            ## 23 rnn_cache4_out : (1, 33, 128) 4224
            ## 24 de_cache1_out : (1, 256, 1, 33) 8448
            ## 25 de_cache2_out : (1, 128, 1, 33) 4224
            ## 26 de_cache3_out : (1, 64, 1, 33) 2112
            ## 27 de_cache4_out : (1, 64, 1, 65) 4160
            ## 28 de_cache5_out : (1, 64, 1, 129) 8256
            ## 29 enh : (1, 257, 1, 2) 514
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

        # Transfer the current frame of noisy data from CPU to CUDA.
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        # Execute inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy cache_out to cache_in within CUDA.
        for i in range(1, 15):
            # print(self.inputs[i].name, self.outputs[i-1].name)
            assert(self.outputs[i-1].size == self.inputs[i].size)
            cuda.memcpy_dtod_async(self.inputs[i].device, self.outputs[i-1].device, self.outputs[i-1].size, self.stream)

        # Transfer the current frame of enhanced data from CUDA to CPU.
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

    x = np.random.randn(1, 257, 1000, 2)
    
    times = np.zeros([x.shape[-2]])
    outputs = np.zeros([1, 257, x.shape[-2], 2])    # [1, F, T, 2]
    for i in range(x.shape[-2]):
        tic = time.perf_counter()
        out_i = model(x[:,:, i:i+1,:])
        toc = time.perf_counter()

        outputs[:,:,i:i+1,:] = out_i
        times[i] = 1000*(toc-tic)

    print("Average Inference Time (ms): ", times.mean())
    print("Maximum Inference Time (ms): ", times.max())
    print("Minimum Inference Time (ms): ", times.min())




    

