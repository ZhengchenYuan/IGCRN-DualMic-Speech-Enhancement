"""Optimizer diagnostic, not a research experiment. Fails unless fixed-batch loss decreases."""
import argparse
import torch
from igcrn.model import IGCRN, spectral_loss
from scripts.train import seed_all
from igcrn.data import Mixtures
p=argparse.ArgumentParser(); p.add_argument('--manifest'); p.add_argument('--steps',type=int,default=60)
a=p.parse_args(); seed_all(2026); torch.set_num_threads(2)
m=IGCRN(width=8,depth=2); opt=torch.optim.Adam(m.parameters(),lr=.003)
if a.manifest: x,y=Mixtures(a.manifest)[0]; x,y=x[None],y[None]
else:
    t=torch.arange(2048)/16000; y=(.2*torch.sin(2*torch.pi*220*t))[None]
    x=torch.stack((y+.03*torch.randn_like(y),y+.03*torch.randn_like(y)),1)
losses=[]
for _ in range(a.steps):
    loss=spectral_loss(m.spectrum(x),m.stft(y)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
print({'initial':losses[0],'final':losses[-1],'steps':a.steps})
assert losses[-1]<losses[0]*.7, 'Tiny-set loss did not decrease by 30%'
