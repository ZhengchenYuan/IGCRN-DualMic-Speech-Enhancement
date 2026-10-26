"""Deterministic first-order image-source RIR; no external RIR dependency."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from math import gcd
from igcrn.data import rows, sha256

def read(path, n, rng):
    x, sr = sf.read(path, always_2d=True); x = x.mean(1)
    d = gcd(sr, 16000); x = resample_poly(x, 16000//d, sr//d)
    if len(x) < n: x = np.tile(x, int(np.ceil(n/len(x))))
    start = int(rng.integers(len(x)-n+1))
    return x[start:start+n], start

def rir(source, mic, room=(5, 5, 3), reflection=.65):
    # Direct path plus six first-order wall images. Linear fractional-delay deposition.
    images = [(np.array(source), 1.)]
    for axis in range(3):
        for wall in (0, room[axis]):
            s = np.array(source); s[axis] = 2*wall-s[axis]
            images.append((s, reflection))
    h = np.zeros(4096)
    for s, gain in images:
        distance = np.linalg.norm(s-mic); delay = distance/343*16000
        k = int(delay); frac = delay-k
        h[k:k+2] += gain/(4*np.pi*distance)*np.array([1-frac, frac])
    return h

def simulate(args):
    catalog = rows(args.catalog)
    speakers = {}; noise_ids = {}
    for r in catalog:
        assert r['kind'] in ('speech', 'noise') and r['split'] in ('train','valid','test')
        assert r.get('license') and r.get('source_url')
        groups = speakers if r['kind'] == 'speech' else noise_ids
        key = r['speaker_id'] if r['kind'] == 'speech' else r['recording_id']
        groups.setdefault(key, set()).add(r['split'])
    assert all(len(v) == 1 for v in list(speakers.values())+list(noise_ids.values())), 'Split leakage'
    assert len({sha256(r['path']) for r in catalog}) == len(catalog), 'Duplicate source files'
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed); n = round(args.seconds*16000)
    for split in ('train', 'valid', 'test'):
        speech = [r for r in catalog if r['split']==split and r['kind']=='speech']
        noise = [r for r in catalog if r['split']==split and r['kind']=='noise']
        if not speech or not noise: raise ValueError(f'Missing sources for {split}')
        manifest = []
        for i in range(args.count if split == 'train' else max(3,args.count//5)):
            s = speech[int(rng.integers(len(speech)))]; q = noise[int(rng.integers(len(noise)))]
            clean, so = read(s['path'], n, rng); ns, no = read(q['path'], n, rng)
            angles = np.arange(-90, 91, 22.5) if split != 'test' else np.arange(-84.375, 90, 11.25)
            a, b = rng.choice(angles, 2, replace=False)
            def position(angle):
                rad = np.deg2rad(angle)
                return [2.5+1.5*np.sin(rad), 2.5+1.5*np.cos(rad), 1.5]
            mics = [np.array([2.49,2.5,1.5]), np.array([2.51,2.5,1.5])]
            hs = [rir(position(a), m) for m in mics]; hn = [rir(position(b), m) for m in mics]
            c = np.stack([fftconvolve(clean,h)[:n] for h in hs], 1)
            v = np.stack([fftconvolve(ns,h)[:n] for h in hn], 1)
            snr = [-3,0,3][i%3]
            if min(np.mean(c[:,0]**2), np.mean(v[:,0]**2)) < 1e-12: raise ValueError('Silent source')
            scale = np.sqrt(np.mean(c[:,0]**2)/np.mean(v[:,0]**2)/10**(snr/10))
            x = c+v*scale; gain = .95/max(np.max(np.abs(x)), np.max(np.abs(c)), 1e-8)
            uid = f'{split}_{i:05d}'; mix = out/f'{uid}_mix.wav'; target = out/f'{uid}_target.wav'
            sf.write(mix, x*gain,16000,subtype='FLOAT'); sf.write(target,c[:,0]*gain,16000,subtype='FLOAT')
            np.savez(out/f'{uid}_rir.npz', speech=hs, noise=hn)
            manifest.append(dict(id=uid,split=split,mixture=str(mix),target=str(target),
                mixture_sha256=sha256(mix),target_sha256=sha256(target),snr_db=snr,
                seed=args.seed,speech=s,noise=q,speech_sha256=sha256(s['path']),noise_sha256=sha256(q['path']),
                speech_offset_16k=so,noise_offset_16k=no,speech_angle=float(a),noise_angle=float(b),
                noise_scale=float(scale),gain=float(gain),samples=n,rir=str(out/f'{uid}_rir.npz')))
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=300); p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--seed',type=int,default=2026); simulate(p.parse_args())
