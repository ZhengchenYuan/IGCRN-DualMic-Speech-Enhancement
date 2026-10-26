"""Paper-inspired IGCRN; tensors use B,C,F,T internally."""
import torch
from torch import nn
from torch.nn import functional as F

class Gated(nn.Module):
    def __init__(self, ci, co, transpose=False, output=False):
        super().__init__()
        conv = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.value = conv(ci, co, (5, 1), padding=(2, 0))
        self.gate = conv(ci, co, (5, 1), padding=(2, 0))
        self.post = nn.Identity() if output else nn.Sequential(nn.BatchNorm2d(co), nn.ELU())
    def forward(self, x):
        return self.post(self.value(x) * self.gate(x).sigmoid())

class Decoder(nn.Module):
    def __init__(self, width, bins, depth):
        super().__init__()
        self.layers = nn.ModuleList([Gated(2*width, width if i < depth-1 else 2,
                                             True, i == depth-1) for i in range(depth)])
        self.linear = nn.Linear(bins, bins)
    def forward(self, x, skips):
        for layer, skip in zip(self.layers, reversed(skips)):
            x = layer(torch.cat((x, skip), 1))
        return self.linear(x.transpose(2, 3)).transpose(2, 3)

class IGCRN(nn.Module):
    def __init__(self, width=64, depth=6, n_fft=512, hop=256):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer('window', torch.hann_window(n_fft).sqrt())
        self.encoder = nn.ModuleList([Gated(4 if i == 0 else width, width) for i in range(depth)])
        self.rnn = nn.LSTM(width, width, 2, batch_first=True, bidirectional=True)
        self.project = nn.Linear(2*width, width)
        self.amplitude = Decoder(width, n_fft//2+1, depth)
        self.phase = Decoder(width, n_fft//2+1, depth)
    def stft(self, x):
        return torch.stft(x, self.n_fft, self.hop, window=self.window,
                          return_complex=True, center=True, pad_mode='constant')
    def istft(self, z, length):
        return torch.istft(z, self.n_fft, self.hop, window=self.window, length=length)
    def spectrum(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError('Expected synchronized B,2,N waveforms')
        b, _, n = waveform.shape
        z = self.stft(waveform.reshape(b*2, n)).reshape(b, 2, self.n_fft//2+1, -1)
        x = torch.cat((z.real, z.imag), 1)
        skips = []
        for layer in self.encoder:
            x = layer(x); skips.append(x)
        _, c, f, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*f, t, c)
        x = self.project(self.rnn(x)[0]).reshape(b, f, t, c).permute(0, 3, 1, 2)
        a = self.amplitude(x, skips)
        magnitude = a[:, 0].sigmoid()*z[:, 0].abs() + F.softplus(a[:, 1])
        p = self.phase(x, skips)
        norm = p.square().sum(1).clamp_min(1e-16).sqrt()
        phase = torch.complex(p[:, 0], p[:, 1]) / norm.clamp_min(1e-8)
        phase = torch.where(norm > 1e-8, phase, torch.ones_like(phase))
        return magnitude*phase
    def forward(self, waveform):
        return self.istft(self.spectrum(waveform), waveform.shape[-1])

def spectral_loss(estimate, target):
    def compress(z):
        a = z.abs().clamp_min(1e-8)
        return a.pow(1/3), z/a*a.pow(1/3)
    a, z = compress(estimate); b, s = compress(target)
    return (a-b).square().mean() + (z.real-s.real).square().mean() + (z.imag-s.imag).square().mean()
