"""
A more elegant implementation of DPCRN.
1.74 GMac, 787.15 k
"""
import torch
import torch.nn as nn


class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel

        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm((width, numUnits), eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm((width, numUnits), eps=1e-8)
    
    def forward(self,x):
        # x: (B, C, T, F)
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
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.channel) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 filter_size,
                 kernel_size,
                 stride_size):
        super().__init__()
        self.N_layers = len(filter_size)

        self.pad_list = nn.ModuleList([])
        self.conv_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        self.act_list = nn.ModuleList([])

        for i in range(self.N_layers):
            Cin = in_channels if i==0 else filter_size[i-1]
            self.pad_list.append(nn.ZeroPad2d([(kernel_size[i][1]-1)//2, (kernel_size[i][1]-1)//2, kernel_size[i][0]-1, 0]))
            self.conv_list.append(nn.Conv2d(Cin, filter_size[i], kernel_size[i], stride_size[i]))
            self.bn_list.append(nn.BatchNorm2d(filter_size[i]))
            self.act_list.append(nn.PReLU())

    def forward(self, x):
        """
        x: (B,C,T,F)
        """
        en_outs = []
        for i in range(self.N_layers):
            x = self.pad_list[i](x)
            x = self.conv_list[i](x)
            x = self.bn_list[i](x)
            x = self.act_list[i](x)
            en_outs.append(x)
            # print(f'en_{i}:', x.shape)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 filter_size,
                 kernel_size,
                 stride_size):
        super().__init__()
        self.N_layers = len(filter_size)

        self.pad_list = nn.ModuleList([])
        self.conv_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        self.act_list = nn.ModuleList([])

        for i in range(self.N_layers-1, -1, -1):
            Cout = out_channels if i==0 else filter_size[i-1]
            act = nn.Identity() if i== 0 else nn.PReLU()
            self.pad_list.append(nn.ZeroPad2d([0, 0, kernel_size[i][0]-1, 0]))
            self.conv_list.append(nn.ConvTranspose2d(filter_size[i]*2, Cout, kernel_size[i], stride_size[i], (kernel_size[i][0]-1, (kernel_size[i][1]-1)//2)))
            self.bn_list.append(nn.BatchNorm2d(Cout))
            self.act_list.append(act)

    def forward(self, x, en_outs):
        """
        x: (B,C,T,F)
        """
        for i in range(self.N_layers):
            x = torch.cat([x, en_outs[self.N_layers-1-i]], dim=1)
            x = self.pad_list[i](x)
            x = self.conv_list[i](x)
            x = self.bn_list[i](x)
            x = self.act_list[i](x)
            # print(f'de_{i}:', x.shape)
        return x


class DPCRN(nn.Module):
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
        self.N_dprnn = N_dprnn
        self.encoder = Encoder(in_channels, filter_size, kernel_size, stride_size)
        
        self.dprnns = nn.ModuleList([])
        for i in range(N_dprnn):
            self.dprnns.append(DPRNN(num_units, width, filter_size[-1]))
            
        self.decoder = Decoder(out_channels, filter_size, kernel_size, stride_size)

    def forward(self, x):
        """
        x: (B,F,T,2), noisy spectrogram, where B is batch size, F is frequency bins, T is time frames, and 2 is R/I components.
        """
        x_ref = x
        x = x.permute(0, 3, 2, 1)     # (B,C,T,F)

        x, en_outs = self.encoder(x)

        for i in range(self.N_dprnn):
            x = self.dprnns[i](x)

        x = self.decoder(x, en_outs)

        m = x.permute(0,3,2,1)

        s_real = x_ref[...,0] * m[...,0] - x_ref[...,1] * m[...,1]
        s_imag = x_ref[...,1] * m[...,0] + x_ref[...,0] * m[...,1]
        s = torch.stack([s_real, s_imag], dim=-1)  # (B,F,T,2)
        
        return s



if __name__ == "__main__":
    model = DPCRN().cuda()

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print(flops, params)

    model = model.cpu().eval()
    x = torch.randn(1, 257, 63, 2)
    y = model(x)
    print(y.shape)