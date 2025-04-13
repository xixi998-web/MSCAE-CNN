import time
import typing as t
from typing import Tuple, Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


# paper title：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
# paper link：https://arxiv.org/pdf/2407.05128

# source code：https://github.com/HZAI-ZJNU/SCSA

class SCSA(nn.Module):
    """A building block combining spatial and channel attention with synergistic effects.

    Integrates grouped depthwise convolutions for spatial processing and cross-channel 
    attention mechanisms. Applies multi-scale context aggregation and efficient attention
    projections to enhance feature representations.

    Args:
        dim (int): Channel dimension of input features
        head_num (int): Number of attention heads
        window_size (int): Window size for downsampling (default: 7)
        group_kernel_sizes (List[int]): Kernel sizes for grouped depthwise convolutions
            (default: [3, 5, 7, 9])
        qkv_bias (bool): Include bias in query/key/value projections (default: False)
        fuse_bn (bool): Fuse batch normalization (unused in current implementation)
        down_sample_mode (str): Downsampling method - 'avg_pool', 'max_pool', or 
            'recombination' (default: 'avg_pool')
        attn_drop_ratio (float): Dropout ratio for attention weights (default: 0.0)
        gate_layer (str): Activation for attention gates - 'sigmoid' or 'softmax'
            (default: 'sigmoid')

    Input:
        x (Tensor): Input feature map with shape (B, C, H, W)

    Output:
        Tensor: Enhanced feature map with same dimensions as input
        Tensor: Attention weights for visualization/debugging
    """
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  
        self.dim = dim  
        self.head_num = head_num  
        self.head_dim = dim // head_num  
        self.scaler = self.head_dim ** -0.5  
        self.group_kernel_sizes = group_kernel_sizes 
        self.window_size = window_size  
        self.qkv_bias = qkv_bias  
        self.fuse_bn = fuse_bn  
        self.down_sample_mode = down_sample_mode  

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  
        self.group_chans = group_chans = self.dim // 4  

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

       
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  
        self.norm_w = nn.GroupNorm(4, dim)  

        self.conv_d = nn.Identity()  
        self.norm = nn.GroupNorm(1, dim)  
        
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  

       
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1)) 
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans 
              
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  

    def forward(self, x: torch.Tensor) -> tuple[Tensor | Any, Any]:
        b, c, h_, w_ = x.size()  
       
        x_h = x.mean(dim=3)  
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  
       
        x_w = x.mean(dim=2)  
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  

     
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  

     
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  

        x = x * x_h_attn * x_w_attn

        y = self.down_func(x)  
        y = self.conv_d(y)  
        _, _, h_, w_ = y.size()  

        y = self.norm(y)  
        q = self.q(y)  
        k = self.k(y)  
        v = self.v(y) 
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        attn = q @ k.transpose(-2, -1) * self.scaler 
        attn = self.attn_drop(attn.softmax(dim=-1)) 
    
        attn = attn @ v  
        
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
       
        attn = attn.mean((2, 3), keepdim=True)  
        attn = self.ca_gate(attn) 

        return attn * x 


if __name__ == "__main__":
    import torch
    import time

    scsa = SCSA(dim=32, head_num=8, window_size=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scsa.to(device)

    input_tensor = torch.rand(1, 32, 256, 256).to(device)
   
    output_tensor = scsa(input_tensor)

    print(f"shape: {output_tensor.shape}")