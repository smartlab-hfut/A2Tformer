import torch
import torch.nn as nn
import torch.nn.functional as F

# === Attention Mechanisms ===

class AutocorrModule(nn.Module):
    def __init__(self, dimension):
        super().__init__()

    def forward(self, x):
        batch, seq_len = x.shape
        x_padded = F.pad(x, (0, seq_len))
        x_fft = torch.fft.fft(x_padded)
        psd = x_fft * torch.conj(x_fft)
        corr = torch.fft.ifft(psd).real
        max_corr = torch.max(corr, dim=1, keepdim=True).values.clamp_min(1e-6)
        corr_norm = corr / max_corr
        return corr_norm[:, :seq_len]


class MixedAttention(nn.Module):
    """
    时域A2T分支 + 多尺度小波分支 + 残差归一化
    """
    def __init__(self, dimension, dropout, num_scales=4, kernel_size=3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension, eps=1e-6)
        self.acf_attention = AutocorrModule(dimension)
        self.pwtm = LearnableWaveletAttention(num_scales=num_scales, kernel_size=kernel_size)
        self.W_acf = nn.Parameter(torch.randn(dimension))  # (S,)

    def forward(self, x):

        # print("x",x.shape)
        # 时域A2T
        acf_attn = self.acf_attention(x)
        # print("acf_attn", acf_attn.shape)
        acf_score = torch.softmax(acf_attn * self.W_acf, dim=-1)
        acf_out = acf_attn * acf_score
        # print("acf_out",acf_out.shape)

        # 多尺度小波特征分支（PWTM）
        wavelet_out = self.pwtm(x)
        wavelet_acf = self.acf_attention(wavelet_out)
        wavelet_score = torch.softmax(wavelet_acf, dim=-1)
        wavelet_feat = wavelet_out * wavelet_score
        # print("wavelet_feat",wavelet_feat.shape)

        # 融合
        attn_out = acf_out + wavelet_feat
        out = self.norm(self.dropout(attn_out))
        return out



class PositionwiseFeedForward(nn.Module):
    def __init__(self, dimension, d_hid, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dimension, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.residual_proj = nn.Linear(dimension, d_hid) if dimension != d_hid else nn.Identity()
        self.norm = nn.LayerNorm(d_hid, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x,return_features=False):
        residual = self.residual_proj(x)
        x_fc1 = self.dropout(self.relu(self.fc1(x)))  # 缩放后的特征
        x_out = self.fc2(x_fc1)
        x_norm = self.norm(x_out + residual)
        if return_features:
            # 返回缩放前和缩放后的特征（不含归一化）
            return x, x_fc1
        return x_norm

# === Transformer Block ===

class TransformerBlock(nn.Module):
    def __init__(self, dimension, d_hid, dropout):
        super().__init__()
        self.attn = MixedAttention(dimension, dropout)
        self.ffn = PositionwiseFeedForward(dimension, d_hid, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

# === Main Model ===

class Transformer(nn.Module):
    def __init__(self, dimension, d_hid, d_inner, n_layers, dropout, num_layers, class_num):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dimension[i], d_hid[i], dropout) for i in range(n_layers)
        ])
        self.init_weights()
        # 若要直接分类，可加头部:
        self.head = nn.Linear(d_hid[-2], class_num)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)
        # 若要分类输出，改成 return self.head(x)



class LearnableWaveletAttention(nn.Module):
    """
    在尺度维度上用标准点积注意力(Q @ K^T)，
    将 (B, S, num_scales) → (B, S)。
    """
    def __init__(self, num_scales=4, kernel_size=33):
        super().__init__()
        self.num_scales  = num_scales
        self.kernel_size = kernel_size

        # 可学习的小波尺度与频率参数
        init_scales = [2.0 * (i + 1) for i in range(num_scales)]
        init_freqs  = [1.0 for _ in range(num_scales)]
        self.scales = nn.Parameter(torch.tensor(init_scales, dtype=torch.float32))  # (num_scales,)
        self.freqs  = nn.Parameter(torch.tensor(init_freqs,  dtype=torch.float32))  # (num_scales,)

        # 下面三层把 每个时刻长为 num_scales 的向量，
        # 映射到 Q、K、V 的 d_k 维空间。示例中我们令 d_k = num_scales。
        # 因此，W_q : (num_scales) → (num_scales)，同理 W_k, W_v。
        self.W_q = nn.Linear(num_scales, num_scales, bias=False)
        self.W_k = nn.Linear(num_scales, num_scales, bias=False)
        self.W_v = nn.Linear(num_scales, num_scales, bias=False)

        # 用于 scale 缩放 scores：d_k = num_scales
        self.scale = float(num_scales) ** 0.5

    def morlet_kernel(self, scale, freq, device):
        """
        生成长度为 self.kernel_size 的 Morlet 小波核，并做归一化。
        """
        t = torch.linspace(
            -self.kernel_size // 2,
             self.kernel_size // 2,
             steps=self.kernel_size,
             device=device
        )
        wave = torch.exp(-0.5 * (t / scale) ** 2) * torch.cos(2 * torch.pi * freq * t / self.kernel_size)
        return wave / (wave.norm() + 1e-6)  # 归一化后返回 shape=(kernel_size,)

    def forward(self, x):
        """
        输入：
          x: Tensor, 形状为 (B, S) 或 (B, S, 1)
        输出：
          out: Tensor, 形状为 (B, S)，
               即每个时刻都用标准的点积注意力把 num_scales 通道融合为标量。
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = x.transpose(1, 2)  # (B, 1, S)
        device = x.device

        coeffs_list = []
        for i in range(self.num_scales):
            kernel = self.morlet_kernel(self.scales[i], self.freqs[i], device=device)
            kernel = kernel.view(1, 1, -1)   # (out_ch=1, in_ch=1, kernel_size)
            padding = self.kernel_size // 2
            c = F.conv1d(x, kernel, padding=padding)  # (B, 1, S)
            coeffs_list.append(c)
        coeffs = torch.cat(coeffs_list, dim=1)       # (B, num_scales, S)

        coeffs = coeffs.permute(0, 2, 1).contiguous()  # (B, S, num_scales)

        Q = self.W_q(coeffs)  # (B, S, num_scales)
        K = self.W_k(coeffs)  # (B, S, num_scales)
        V = self.W_v(coeffs)  # (B, S, num_scales)

        scores = (Q.unsqueeze(-1) * K.unsqueeze(-2))  # (B, S, num_scales, num_scales)

        alpha = F.softmax(scores / self.scale, dim=-1)  # (B, S, num_scales, num_scales)

        out_vec = torch.matmul(alpha, V.unsqueeze(-1)).squeeze(-1)

        out = out_vec.mean(dim=-1)  # (B, S)

        return out












