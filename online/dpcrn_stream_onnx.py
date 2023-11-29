"""
A more elegant implementation of DPCRN.
1.74 GMac, 787.15 k
"""
import torch
import torch.nn as nn
from modules.convolution import StreamConv2d, StreamConvTranspose2d
from modules.convert import convert_to_stream


class StreamDPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super().__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm((width, numUnits), eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first=True, bidirectional=False)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm((width, numUnits), eps=1e-8)
    
    def forward(self, x, h_cache, c_cache):
        """
        x:       (B, C, T=1, F)
        h_cache: (1, F, C), hidden cache for inter RNN.
        c_cache: (1, F, C), cell cache for inter RNN.
        """
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.channel) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x, (h_cache, c_cache) = self.inter_rnn(inter_x, (h_cache, c_cache))  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.channel) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out, h_cache, c_cache


class StreamEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 filter_size,
                 kernel_size,
                 stride_size):
        super().__init__()
        self.N_layers = len(filter_size)

        self.conv_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        self.act_list = nn.ModuleList([])

        for i in range(self.N_layers):
            Cin = in_channels if i==0 else filter_size[i-1]
            self.conv_list.append(StreamConv2d(Cin, filter_size[i], kernel_size[i], stride_size[i], (0, (kernel_size[i][1]-1)//2)))
            self.bn_list.append(nn.BatchNorm2d(filter_size[i]))
            self.act_list.append(nn.PReLU())

    def forward(self, x, en_caches):
        """
        x: (B,C,1,F).
        en_caches: A list of cache (B,C,1,F) for each conv layer in encoder. 
        """
        en_outs = []
        for i in range(self.N_layers):
            x, en_caches[i] = self.conv_list[i](x, en_caches[i])
            x = self.bn_list[i](x)
            x = self.act_list[i](x)
            en_outs.append(x)
            # print(f'en_{i}:', x.shape)
        return x, en_outs, en_caches


class StreamDecoder(nn.Module):
    def __init__(self,
                 out_channels,
                 filter_size,
                 kernel_size,
                 stride_size):
        super().__init__()
        self.N_layers = len(filter_size)

        self.conv_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        self.act_list = nn.ModuleList([])

        for i in range(self.N_layers-1, -1, -1):
            Cout = out_channels if i==0 else filter_size[i-1]
            act = nn.Identity() if i== 0 else nn.PReLU()
            self.conv_list.append(StreamConvTranspose2d(filter_size[i]*2, Cout, kernel_size[i], stride_size[i], (0, (kernel_size[i][1]-1)//2)))
            self.bn_list.append(nn.BatchNorm2d(Cout))
            self.act_list.append(act)

    def forward(self, x, en_outs, de_caches):
        """
        x: (B,C,T,F)
        de_caches: A list of cache (B,C,T,F) for each conv layer in decoder. 
        """
        for i in range(self.N_layers):
            x = torch.cat([x, en_outs[self.N_layers-1-i]], dim=1)
            x, de_caches[i] = self.conv_list[i](x, de_caches[i])
            x = self.bn_list[i](x)
            x = self.act_list[i](x)
            # print(f'de_{i}:', x.shape)
        return x, de_caches


class StreamDPCRN(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 filter_size=[32,32,32,64,128],
                 kernel_size=[(2,5), (2,3), (2,3), (2,3), (2,3)],
                 stride_size=[(1,2), (1,2), (1,2), (1,1), (1,1)],
                 N_dprnn=2,
                 num_units=128,
                 width=33,
                 **kwargs):
        super().__init__()
        self.encoder = StreamEncoder(in_channels, filter_size, kernel_size, stride_size)
        
        self.dprnns = nn.ModuleList([])
        for i in range(N_dprnn):
            self.dprnns.append(StreamDPRNN(num_units, width, filter_size[-1]))
            
        self.decoder = StreamDecoder(out_channels, filter_size, kernel_size, stride_size)

    def forward(self, x,
                en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,
                rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,
                de_cache1, de_cache2, de_cache3, de_cache4, de_cache5):
        """
        When export to onnx, the inputs should be specific tensors.
        x: (B,F,T,2)
        cachees: ...
        """
        x_ref = x
        x = x.permute(0, 3, 2, 1)     # (B,C,T,F)

        x, en_outs, [en_cache1, en_cache2, en_cache3, en_cache4, en_cache5] = self.encoder(x, [en_cache1, en_cache2, en_cache3, en_cache4, en_cache5])

        x, rnn_cache1, rnn_cache2 = self.dprnns[0](x, rnn_cache1, rnn_cache2)
        x, rnn_cache3, rnn_cache4 = self.dprnns[1](x, rnn_cache3, rnn_cache4)

        x, [de_cache1, de_cache2, de_cache3, de_cache4, de_cache5] = self.decoder(x, en_outs, [de_cache1, de_cache2, de_cache3, de_cache4, de_cache5])

        m = x.permute(0,3,2,1)

        s_real = x_ref[...,0] * m[...,0] - x_ref[...,1] * m[...,1]
        s_imag = x_ref[...,1] * m[...,0] + x_ref[...,0] * m[...,1]
        s = torch.stack([s_real, s_imag], dim=-1)  # (B,F,T,2)
        
        return s, en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,\
                rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,\
                de_cache1, de_cache2, de_cache3, de_cache4, de_cache5



if __name__ == "__main__":
    from dpcrn import DPCRN
    model = DPCRN().eval()
    model_stream = StreamDPCRN().eval()
    convert_to_stream(model_stream, model)

    x = torch.randn(1, 257, 1000, 2)
    en_cache1 = torch.zeros(1, 2,  1, 257)
    en_cache2 = torch.zeros(1, 32, 1, 129)
    en_cache3 = torch.zeros(1, 32, 1, 65)
    en_cache4 = torch.zeros(1, 32, 1, 33)
    en_cache5 = torch.zeros(1, 64,1, 33)

    rnn_cache1 = torch.zeros(1, 33, 128)
    rnn_cache2 = torch.zeros(1, 33, 128)
    rnn_cache3 = torch.zeros(1, 33, 128)
    rnn_cache4 = torch.zeros(1, 33, 128)

    de_cache1 = torch.zeros(1, 256,1, 33)
    de_cache2 = torch.zeros(1, 128,1, 33)
    de_cache3 = torch.zeros(1, 64, 1, 33)
    de_cache4 = torch.zeros(1, 64, 1, 65)
    de_cache5 = torch.zeros(1, 64, 1, 129)
    
    y1 = []
    for i in range(x.shape[-2]):
        yi, en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,\
            rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,\
            de_cache1, de_cache2, de_cache3, de_cache4, de_cache5 = \
                    model_stream(x[:,:,i:i+1,:], en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,
                                            rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,
                                            de_cache1, de_cache2, de_cache3, de_cache4, de_cache5)
        y1.append(yi)
    y1 = torch.cat(y1, dim=2)
    
    ## check streaming errors
    y = model(x)
    print((y-y1).abs().max())


    import time
    import onnx
    import onnxruntime
    import numpy as np
    from onnxsim import simplify
    ## convert to onnx
    file = 'models/dpcrn.onnx'
    device = torch.device('cpu')
    input = torch.randn(1, 257, 1, 2, device=device)
    torch.onnx.export(model_stream,
                    (input, en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,
                        rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,
                        de_cache1, de_cache2, de_cache3, de_cache4, de_cache5),
                    file,
                    input_names = ['mix', 'en_cache1', 'en_cache2', 'en_cache3', 'en_cache4', 'en_cache5',
                                   'rnn_cache1', 'rnn_cache2', 'rnn_cache3', 'rnn_cache4',
                                    'de_cache1', 'de_cache2', 'de_cache3', 'de_cache4', 'de_cache5'],
                    output_names = ['enh', 'en_cache1_out', 'en_cache2_out', 'en_cache3_out', 'en_cache4_out', 'en_cache5_out',
                                   'rnn_cache1_out', 'rnn_cache2_out', 'rnn_cache3_out', 'rnn_cache4_out',
                                    'de_cache1_out', 'de_cache2_out', 'de_cache3_out', 'de_cache4_out', 'de_cache5_out'],
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
    input = x.cpu().detach().numpy()
    en_cache1 = torch.zeros(1, 2,  1, 257).numpy()
    en_cache2 = torch.zeros(1, 32, 1, 129).numpy()
    en_cache3 = torch.zeros(1, 32, 1, 65).numpy()
    en_cache4 = torch.zeros(1, 32, 1, 33).numpy()
    en_cache5 = torch.zeros(1, 64,1, 33).numpy()

    rnn_cache1 = torch.zeros(1, 33, 128).numpy()
    rnn_cache2 = torch.zeros(1, 33, 128).numpy()
    rnn_cache3 = torch.zeros(1, 33, 128).numpy()
    rnn_cache4 = torch.zeros(1, 33, 128).numpy()

    de_cache1 = torch.zeros(1, 256,1, 33).numpy()
    de_cache2 = torch.zeros(1, 128,1, 33).numpy()
    de_cache3 = torch.zeros(1, 64, 1, 33).numpy()
    de_cache4 = torch.zeros(1, 64, 1, 65).numpy()
    de_cache5 = torch.zeros(1, 64, 1, 129).numpy()

    T_list = []
    outputs = []

    for i in range(input.shape[-2]):
        tic = time.perf_counter()
        
        out_i, en_cache1, en_cache2, en_cache3, en_cache4, en_cache5,\
            rnn_cache1, rnn_cache2, rnn_cache3, rnn_cache4,\
            de_cache1, de_cache2, de_cache3, de_cache4, de_cache5 \
                = session.run([], {'mix': input[..., i:i+1, :],
                                    'en_cache1': en_cache1, 'en_cache2':en_cache2, 'en_cache3':en_cache3, 'en_cache4':en_cache4, 'en_cache5':en_cache5,
                                   'rnn_cache1':rnn_cache1, 'rnn_cache2':rnn_cache2, 'rnn_cache3':rnn_cache3, 'rnn_cache4':rnn_cache4,
                                    'de_cache1':de_cache1, 'de_cache2':de_cache2, 'de_cache3':de_cache3, 'de_cache4':de_cache4, 'de_cache5':de_cache5})

        toc = time.perf_counter()
        T_list.append(toc-tic)
        outputs.append(out_i)
    outputs = np.concatenate(outputs, axis=2)
    ## check onnx errors
    print(np.abs(outputs - y.detach().numpy()).max())
    
    ## evaluate inference speed
    print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))