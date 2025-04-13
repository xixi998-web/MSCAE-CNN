import torch
import torch.nn as nn

class FrequencyLayer(nn.Module):
    """Frequency domain processing layer combining FFT-based filtering with learnable parameters.
    
    Args:
        h (int): Height of input feature maps
        w (int): Width of input feature maps

    Implements:
        - FFT-based low/high frequency decomposition
        - Learnable frequency weighting (sqrt_beta parameter)
        - Residual connection with LayerNorm
    """
    def __init__(self, h, w):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(0.5)
        self.LayerNorm = nn.LayerNorm(h * w, eps=1e-12)
 
        self.c_h = h // 2 + 1  
        self.c_w = w // 2 + 1 
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, h, w))

    def forward(self, input_tensor):
        """Process input through frequency decomposition and enhancement.
        
        1. Compute FFT and split into low/high frequency components
        2. Apply learnable weighting to high frequencies
        3. Reconstruct with residual connection
        """
        batch, c, h, w = input_tensor.shape

        x = torch.fft.rfft2(input_tensor, dim=(2, 3), norm='ortho')

        low_pass = x.clone()
        low_pass[:, :, self.c_h:, :] = 0  
        low_pass[:, :, :, self.c_w:] = 0 
        low_pass = torch.fft.irfft2(low_pass, s=(h, w), dim=(2, 3), norm='ortho')

        high_pass = input_tensor - low_pass

        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states.view(batch, -1)).view(batch, c, h, w)
        
        hidden_states = hidden_states + input_tensor

        return hidden_states