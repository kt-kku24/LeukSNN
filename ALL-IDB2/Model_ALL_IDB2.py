import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Model for RM-ResNet

thresh = 0.5  # neuronal threshold
decay = 0.5  # decay constants
num_classes = 1
time_window = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        
        return out.view(x.shape[0], x.shape[1], 1, 1) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        o = torch.cat([avgout, maxout], dim=1)
        o = self.conv(o)


        return self.sigmoid(o) * x



class ATan(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out
        

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = (alpha/ 2 / (1 + (torch.pi / 2 * alpha * input_).pow_(2))* grad_input)
        return grad, None


act_fun = ATan.apply

class mem_update(nn.Module):

    def __init__(self):
        super(mem_update, self).__init__()
        self.mem = torch.zeros(1, device='cuda')
        
    def forward(self, x, time_step):

        if(time_step == 0):
            self.mem = torch.zeros_like(x, device=x.device)
        
            
        self.mem = self.mem * decay + x
            
        spike = act_fun(self.mem - thresh)

        self.mem = self.mem * (1 - spike.detach())

        return spike



class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, padding, add_res=False, add_attn=False):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels, in_channels, k_size, stride=stride, padding=padding, groups=in_channels)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.lif2 = mem_update()
        self.conv_s = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.lif3 = mem_update()
        self.out_channels = out_channels
        self.k_size = k_size
        self.padding = padding
        self.stride = stride
        self.res_conv = nn.Sequential()
        self.c_attn = nn.Sequential()
        self.s_attn = nn.Sequential()
        self.add_attn = add_attn
        self.add_res = add_res
        if add_attn:
            self.s_attn = nn.Sequential(SpatialAttention(kernel_size=7))
            self.c_attn = nn.Sequential(ChannelAttention(out_channels, 4))
        if add_res:
            self.res_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))
        

    def forward(self, x, time_step):
        
        o = self.conv(x)
        o = self.bn1(o)
        o = self.s_attn(o)
        o = self.lif2(o, time_step)
        spk1 = o.detach()
        
        o = self.conv_s(o)
        o = self.bn2(o)
        o = self.c_attn(o)
        
        if self.add_res:
            o = self.res_conv(x) + o

        
        o = self.lif3(o, time_step)
        spk2 = o.detach()

        return o, spk1, spk2
        


class Model(nn.Module):
    # Channel:
    def __init__(self, num_classes=1):
        super().__init__()
        k = 1
       
        self.conv = DSConv(3, 32, 3, 2, 1, add_res=True, add_attn=True)
        
        self.conv1 = DSConv(32, 64, 3, 2, 1, add_res=True, add_attn=True)

        self.conv2 = DSConv(64, 128, 3, 2, 1, add_res=True, add_attn=True)
        
        self.conv3 = DSConv(128, 256, 3, 2, 1, add_res=True, add_attn=True)

        self.conv4 = DSConv(256, 512, 3, 2, 1, add_res=True, add_attn=True)

        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        b, c, h, w = x.shape
        final_output = torch.zeros((time_window, b, 512, 7, 7), device=x.device)
        spk_rates = torch.zeros((10, time_window), device=x.device)
        for t in range(time_window):
            
            
            output, spk1, spk2 = self.conv(x, t)
            spk_rates[0, t] = spk1.sum()/(spk1.shape[0] * spk1.shape[1] * spk1.shape[2] * spk1.shape[3])
            spk_rates[1, t] = spk2.sum()/(spk2.shape[0] * spk2.shape[1] * spk2.shape[2] * spk2.shape[3])
            
            output, spk1, spk2 = self.conv1(output, t)
            spk_rates[2, t] = spk1.sum()/(spk1.shape[0] * spk1.shape[1] * spk1.shape[2] * spk1.shape[3])
            spk_rates[3, t] = spk2.sum()/(spk2.shape[0] * spk2.shape[1] * spk2.shape[2] * spk2.shape[3])
            
            output, spk1, spk2 = self.conv2(output, t)
            spk_rates[4, t] = spk1.sum()/(spk1.shape[0] * spk1.shape[1] * spk1.shape[2] * spk1.shape[3])
            spk_rates[5, t] = spk2.sum()/(spk2.shape[0] * spk2.shape[1] * spk2.shape[2] * spk2.shape[3])
            
            output, spk1, spk2 = self.conv3(output, t)
            spk_rates[6, t] = spk1.sum()/(spk1.shape[0] * spk1.shape[1] * spk1.shape[2] * spk1.shape[3])
            spk_rates[7, t] = spk2.sum()/(spk2.shape[0] * spk2.shape[1] * spk2.shape[2] * spk2.shape[3])
            
            output, spk1, spk2 = self.conv4(output, t)
            spk_rates[8, t] = spk1.sum()/(spk1.shape[0] * spk1.shape[1] * spk1.shape[2] * spk1.shape[3])
            spk_rates[9, t] = spk2.sum()/(spk2.shape[0] * spk2.shape[1] * spk2.shape[2] * spk2.shape[3])

            final_output[t] = output
        
        output = F.adaptive_avg_pool3d(final_output, (None, 1, 1))
        
        output = output.view(output.size()[0], output.size()[1], -1)

        output = output.sum(dim=0) / output.size()[0]
        
        output = self.fc2(output)
        
        return output, spk_rates