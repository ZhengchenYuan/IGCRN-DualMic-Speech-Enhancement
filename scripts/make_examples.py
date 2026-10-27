"""Original synthetic diagnostic audio: not speech and not model outputs."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from scripts.simulate import rir
from igcrn.data import sha256
out=Path('examples/diagnostics'); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(2026); sr=16000
t=np.arange(sr*2)/sr
s=.15*np.sin(2*np.pi*(180*t+25*t*t))*(.5+.5*np.sin(2*np.pi*3*t))
n=rng.normal(0,.06,len(t)); mics=[np.array([2.49,2.5,1.5]),np.array([2.51,2.5,1.5])]
c=np.stack([fftconvolve(s,rir([2,4,1.5],m))[:len(s)] for m in mics],1)
v=np.stack([fftconvolve(n,rir([4,2,1.5],m))[:len(s)] for m in mics],1)
x=c+v; items={'01_dry_chirp':s,'02_reverberant_target_mic1':c[:,0],
 '03_synchronized_dual_mixture':x,'04_duplicated_mic1_mixture':x[:,[0,0]]}
manifest=[]
for name,wave in items.items():
    path=out/(name+'.wav'); sf.write(path,wave,sr,subtype='FLOAT')
    manifest.append(dict(path=str(path),sha256=sha256(path),seed=2026,description='Synthetic diagnostic; not speech or enhancement result'))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
