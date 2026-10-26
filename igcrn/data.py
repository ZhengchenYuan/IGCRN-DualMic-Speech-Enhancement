import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

class Mixtures(Dataset):
    def __init__(self, manifest, mode='dual'):
        self.items = rows(manifest); self.mode = mode
        if not self.items: raise ValueError('Empty manifest')
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]
        x, sr = sf.read(r['mixture'], dtype='float32', always_2d=True)
        y, sy = sf.read(r['target'], dtype='float32')
        assert sr == sy == 16000 and x.shape == (len(y), 2) and y.ndim == 1
        assert sha256(r['mixture']) == r['mixture_sha256'] and sha256(r['target']) == r['target_sha256']
        if self.mode == 'duplicate': x = x[:, [0, 0]]
        return torch.from_numpy(x.T.copy()), torch.from_numpy(y.copy())
