import os
import math
import torch.nn.functional as F

from SMSA import *
from frequencyLayer import FrequencyLayer

class Block(nn.Module):
    """A fundamental building block combining frequency processing and attention mechanisms.
    
    Args:
        h (int): Input feature map height
        w (int): Input feature map width
        dim (int): Channel dimension of input
        head_num (int): Number of attention heads
        attn_drop (float): Dropout ratio for attention weights
        resid_drop (float): Dropout ratio for residual connections
    """
    def __init__(self, h, w, dim, head_num, attn_drop, resid_drop):
        super().__init__()
        # Frequency domain processing layer
        self.filter_layer = FrequencyLayer(h, w, dim)
        
        # Learnable blending parameter between frequency and spatial features
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.7)  
        
        # Spatial-Channel Self-Attention module
        self.attn = SCSA(dim, head_num, window_size=7, attn_drop_ratio=attn_drop) 
        
        # Multi-Layer Perceptron with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(h * w, h * w),
            nn.GELU(),
            nn.Linear(h * w, h * w),
            nn.Dropout(resid_drop),
        )
        
    def forward(self, x):
        """Process input through frequency and attention paths, then combine them."""
        batch, c, h, w = x.shape
        
        # Frequency domain processing
        dsp = self.filter_layer(x)
        
        # Spatial attention processing
        gsp = x + self.attn(x)
        gsp = F.relu(gsp)

        # Adaptive blending of frequency and spatial features
        alpha = torch.sigmoid(self.alpha)
        hidden_states = alpha * dsp + (1 - alpha) * gsp
        hidden_states = F.relu(hidden_states)

        # MLP processing with residual connection
        hidden_states = hidden_states.view(batch, c, h * w)
        output = hidden_states + self.mlp(hidden_states)
        output = F.relu(output)
        output = output.view(batch, c, h, w)
        return output

class Ups(nn.Module):
    """Upsampling block using PixelShuffle with learnable parameters.
    
    Args:
        in_channels (int): Number of input channels
        device (str): Computation device (default: 'cuda')
    """
    def __init__(self, in_channels, device='cuda'):
        super().__init__()
        self.device = device
        # Convolution before upsampling
        self.conv_F = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1).to(self.device)
        
    def forward(self, input_tensor):
        """Upsample input using PixelShuffle operation."""
        Conv_F = self.conv_F(input_tensor)
        # Increase spatial resolution by factor of 2
        output = F.pixel_shuffle(Conv_F, upscale_factor=2) 
        output = F.relu(output)
        return output

class Downs(nn.Module):
    """Downsampling block using PixelUnshuffle with learnable parameters.
    
    Args:
        in_channels (int): Number of input channels
        device (str): Computation device (default: 'cuda')
    """
    def __init__(self, in_channels, device='cuda'):
        super().__init__()
        self.device = device
        # Convolution before downsampling
        self.conv_F = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1).to(self.device)
        
    def forward(self, input_tensor):
        """Downsample input using PixelUnshuffle operation."""
        Conv_F = self.conv_F(input_tensor)
        # Decrease spatial resolution by factor of 2
        output = F.pixel_unshuffle(Conv_F, downscale_factor=2)
        output = F.relu(output)
        return output

class Residual(nn.Module):
    """Residual block with scaled residual connection for stable training.
    
    Args:
        filters (int): Number of input/output channels
        kernel_size (int): Convolution kernel size
    """
    def __init__(self, filters, kernel_size):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=1)

    def forward(self, input_tensor):
        """Residual connection with 0.1 scaling factor."""
        x1 = F.relu(self.conv1(input_tensor))
        x2 = self.conv2(x1)
        # Scaled residual connection
        x2 = x2 * 0.1
        output = input_tensor + x2
        return output

class MSSANet(nn.Module):
    """Multi-Scale Spatial-Spectral Attention Network for super-resolution.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        basenum (int): Base number of filters for channel scaling
        upscale_factor (int): Upsampling ratio
        attn_drop (float): Attention dropout ratio
        head_num (int): Number of attention heads
        n_layer (int): Number of attention blocks
        size (int): Input spatial dimension
        use_emb_layer (bool): Whether to use month embedding
        use_self_attention (bool): Whether to use attention blocks
        use_fourier (bool): Whether to use frequency processing
        resid_drop (float): Residual dropout ratio
        device (str): Computation device (default: 'cuda')
    """
    def __init__(self, in_channels, out_channels, basenum, upscale_factor, attn_drop, head_num, n_layer, size, 
                 use_emb_layer=False, use_self_attention=False, use_fourier=False, resid_drop=0.1, device='cuda'):
        super(MSSANet, self).__init__()
        # Initialize network parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.use_emb_layer = use_emb_layer
        self.use_self = use_self_attention
        self.use_fourier = use_fourier
        self.upscale_factor = upscale_factor
        self.size = size

        # Month embedding layer for temporal information
        if self.use_emb_layer:
            self.emb_layer = nn.Embedding(12, 512*512)
            nn.init.normal_(self.emb_layer.weight, mean=0.0, std=0.1)

        # Multi-scale feature extraction layers
        self.convMultiScale1 = nn.Conv2d(self.in_channels, basenum, kernel_size=3, padding=1).to(self.device)
        self.convMultiScale2 = nn.Conv2d(basenum, basenum, kernel_size=3, padding=1).to(self.device)
        self.convMultiScale3 = nn.Conv2d(basenum, basenum, kernel_size=3, padding=1).to(self.device)

        # Downsampling blocks
        self.downs1 = Downs(in_channels=3 * basenum, device=self.device)
        self.downs2 = Downs(in_channels=3 * basenum * 2, device=self.device)
        self.downs3 = Downs(in_channels=3 * basenum * 4, device=self.device)

        # Attention blocks (either pure attention or hybrid frequency-attention)
        if self.use_self and not self.use_fourier:
            self.Blocks = nn.Sequential(*[
                SCSA(3 * basenum * 8, self.head_num, window_size=7, attn_drop_ratio=self.attn_drop) 
                for _ in range(self.n_layer)]).to(self.device)
        elif self.use_self and self.use_fourier:
            self.Blocks = nn.Sequential(*[
                Block(self.size//8*3, self.size//8*3, 3 * basenum * 8, self.head_num, self.attn_drop, self.resid_drop)
                for _ in range(self.n_layer)]).to(self.device)

        # Upsampling blocks
        self.ups1 = Ups(in_channels=3 * basenum * 8, device=self.device)
        self.ups2 = Ups(in_channels=3 * basenum * 4, device=self.device)
        self.ups3 = Ups(in_channels=3 * basenum * 2, device=self.device)

        # Residual blocks for feature refinement
        self.residual_blocks1 = nn.Sequential(
            Residual(3 * basenum * 4, 3),
            Residual(3 * basenum * 4, 3),
            Residual(3 * basenum * 4, 3),
            Residual(3 * basenum * 4, 3)
        ).to(self.device)

        self.residual_blocks2 = nn.Sequential(
            Residual(3 * basenum * 2, 3),
            Residual(3 * basenum * 2, 3),
            Residual(3 * basenum * 2, 3),
            Residual(3 * basenum * 2, 3)
        ).to(self.device)

        # Final output convolution
        self.conv7 = nn.Conv2d(3 * basenum, self.out_channels, kernel_size=3, padding=1).to(self.device)

    def upsample(self, x, scale=3):
        """Bicubic upsampling for initial feature enlargement."""
        return F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False)

    def forward(self, x, month=None):
        """Network forward pass with optional month embedding."""
        # Process month embedding if enabled
        if self.use_emb_layer:
            month = month - 1
            size = min(h, w)
            month_emb = self.emb_layer(month)
            month_emb = month_emb[:, :size * size]
            month_emb = month_emb.reshape(b, 1, size, size)
            month_emb_normalized = month_emb / 11.0
            x = x + month_emb_normalized

        # Initial upsampling
        x = self.upsample(x, scale=self.upscale_factor)

        # Multi-scale feature extraction
        convMS_1 = F.relu(self.convMultiScale1(x))
        convMS_2 = F.relu(self.convMultiScale2(convMS_1))
        convMS_3 = F.relu(self.convMultiScale3(convMS_2))
        convMS_all = F.relu(torch.cat((convMS_1, convMS_2, convMS_3), dim=1))

        # Downsampling path
        conv1 = F.relu(self.conv1(convMS_all))
        conv1_ = self.downs1(conv1)
        conv2 = F.relu(self.conv2(conv1_))
        conv2_ = self.downs2(conv2)
        conv3 = F.relu(self.conv3(conv2_))
        conv3_ = self.downs3(conv3)

        # Process through attention blocks
        fea = self.Blocks(conv3_) if self.use_self else conv3_

        # Upsampling path with residual connections
        conv4 = self.ups1(fea)
        conv4_ = F.relu(self.conv4(torch.cat((conv2_, conv4), dim=1)))
        conv4_res = self.residual_blocks1(conv4_)

        conv5 = self.ups2(conv4_res)
        conv5_ = F.relu(self.conv5(torch.cat((conv1_, conv5), dim=1)))
        conv5_res = self.residual_blocks2(conv5_)

        conv6 = self.ups3(conv5_res)
        conv6_ = F.relu(self.conv6(torch.cat((convMS_all, conv6), dim=1)))

        # Final output
        res = F.relu(self.conv7(conv6_))
        return res

if __name__ == '__main__':
    # Test network with sample input
    model = MSSANet(in_channels=6, out_channels=10, basenum=16, upscale_factor=3, 
                    attn_drop=0.5, size=80, head_num=8, n_layer=1, 
                    use_emb_layer=True, use_self_attention=True, use_fourier=True, 
                    resid_drop=0.5).cuda()

    x = torch.rand(1, 6, 80, 80).cuda()
    y = model(x, month=torch.tensor([5], device='cuda'))
    print(y.shape)  # Expected output shape: (1, 10, 240, 240)