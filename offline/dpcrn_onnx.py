"""
A more elegant implementation of DPCRN.
1.74 GMac, 787.15 k
"""
import torch
import time
import onnx
import onnxruntime
import numpy as np
from onnxsim import simplify
from dpcrn import DPCRN


## load model
model = DPCRN().eval()  # remember to set `eval` mode!

## convert to onnx
file = 'models/dpcrn.onnx'
device = torch.device('cpu')

time_len = 9  # set the length to 9 s
frame_num = time_len * 16000 // 256 + 1  # compute frame numbers, fs=16000, hop_size=256
x = torch.randn(1, 257, frame_num, 2, device=device)  

torch.onnx.export(model,
                (x,),
                file,
                input_names = ['mix'],
                output_names = ['enh'],
                opset_version=11,
                verbose = False)

onnx_model = onnx.load(file)
onnx.checker.check_model(onnx_model)

model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, file.split('.onnx')[0] + '_simple.onnx')

## run onnx model
# session = onnxruntime.InferenceSession(file, None, providers=['CUDAExecutionProvider'])
session = onnxruntime.InferenceSession(file.split('.onnx')[0]+'_simple.onnx', None, providers=['CPUExecutionProvider'])
inputs = x.cpu().detach().numpy()

## execute inference
outputs = session.run([], {'mix': inputs})

## check error
y = model(x)
diff = outputs - y.detach().numpy()
print(">>> The maximum numerical error:", np.abs(diff).max())

## test inference speed
T_list = []
for i in range(1000):  
    tic = time.perf_counter()
    outputs = session.run([], {'mix': inputs})
    toc = time.perf_counter()
    T_list.append((toc-tic) / time_len)
print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))
