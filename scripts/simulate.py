"""Deterministic first-order image-source RIR; no external RIR dependency."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from scipy.io import wavfile
from math import gcd
from igcrn.data import rows, sha256

MAX_ATTEMPTS = 20
MIN_SOURCE_ENERGY = 1e-12


class SilentSegmentError(ValueError):
    """An empty recording cannot supply a usable segment."""


def read(path, n, rng):
    x, sr = sf.read(path, always_2d=True); x = x.mean(1)
    if not len(x):
        raise SilentSegmentError(f'Empty source recording: {path}')
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
    out = Path(args.output).resolve()
    if args.count <= 0 or not np.isfinite(args.seconds) or round(args.seconds*16000) <= 0:
        raise ValueError('Sample count and duration must be positive')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Output must be new or empty: {out}. Use a new directory to preserve frozen data.')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage audio and all manifests together. A single directory rename publishes
    # the complete run; even SIGKILL can leave only a visibly incomplete sibling.
    with tempfile.TemporaryDirectory(prefix=f'.{out.name}.incomplete-', dir=out.parent) as temporary:
        staging = Path(temporary)
        _simulate_splits(args, catalog, out, staging)
        staging.replace(out)


def _simulate_splits(args, catalog, out, staging):
    rng = np.random.default_rng(args.seed); n = round(args.seconds*16000)
    for split in ('train', 'valid', 'test'):
        speech = [r for r in catalog if r['split']==split and r['kind']=='speech']
        noise = [r for r in catalog if r['split']==split and r['kind']=='noise']
        if not speech or not noise: raise ValueError(f'Missing sources for {split}')
        manifest = []
        for i in range(args.count if split == 'train' else max(3,args.count//5)):
            # Draw from this split only, and consume the same seeded generator
            # for every attempt. Successful first attempts keep the old RNG order.
            for attempt in range(1, MAX_ATTEMPTS + 1):
                s = speech[int(rng.integers(len(speech)))]; q = noise[int(rng.integers(len(noise)))]
                try:
                    clean, so = read(s['path'], n, rng); ns, no = read(q['path'], n, rng)
                except SilentSegmentError as exc:
                    reason = str(exc)
                    continue
                angles = np.arange(-90, 91, 22.5) if split != 'test' else np.arange(-84.375, 90, 11.25)
                a, b = rng.choice(angles, 2, replace=False)
                def position(angle):
                    rad = np.deg2rad(angle)
                    return [2.5+1.5*np.sin(rad), 2.5+1.5*np.cos(rad), 1.5]
                mics = [np.array([2.49,2.5,1.5]), np.array([2.51,2.5,1.5])]
                hs = [rir(position(a), m) for m in mics]; hn = [rir(position(b), m) for m in mics]
                c = np.stack([fftconvolve(clean,h)[:n] for h in hs], 1)
                v = np.stack([fftconvolve(ns,h)[:n] for h in hn], 1)
                speech_energy = float(np.mean(c[:,0]**2))
                noise_energy = float(np.mean(v[:,0]**2))
                if (np.isfinite(speech_energy) and np.isfinite(noise_energy)
                        and min(speech_energy, noise_energy) >= MIN_SOURCE_ENERGY):
                    break
                reason = f'Mic1 speech mean-square={speech_energy:.6g}, noise mean-square={noise_energy:.6g}'
            else:
                raise ValueError(
                    f'Silent source retry exhaustion: split={split}, sample_index={i} (zero-based), '
                    f'attempts={MAX_ATTEMPTS}; {reason}; '
                    f'speech={s["path"]}, noise={q["path"]}. '
                    f'Need finite reverberant Mic1 mean-square >= {MIN_SOURCE_ENERGY:g} for both sources. '
                    'Check the split recordings or segment duration; no dataset was published.')
            snr = [-3,0,3][i%3]
            scale = np.sqrt(np.mean(c[:,0]**2)/np.mean(v[:,0]**2)/10**(snr/10))
            x = c+v*scale; gain = .95/max(np.max(np.abs(x)), np.max(np.abs(c)), 1e-8)
            uid = f'{split}_{i:05d}'; mix = out/f'{uid}_mix.wav'; target = out/f'{uid}_target.wav'
            # libsndfile FLOAT WAVs include a wall-clock timestamp in PEAK.
            # SciPy writes IEEE float32 WAVs without that optional chunk; keep
            # the same sample precision and hash the entire deterministic file.
            wavfile.write(staging/mix.name, 16000, (x*gain).astype(np.float32))
            wavfile.write(staging/target.name, 16000, (c[:,0]*gain).astype(np.float32))
            np.savez(staging/f'{uid}_rir.npz', speech=hs, noise=hn)
            manifest.append(dict(id=uid,split=split,mixture=str(mix),target=str(target),
                mixture_sha256=sha256(staging/mix.name),target_sha256=sha256(staging/target.name),snr_db=snr,
                retry_count=attempt-1,
                seed=args.seed,speech=s,noise=q,speech_sha256=sha256(s['path']),noise_sha256=sha256(q['path']),
                speech_offset_16k=so,noise_offset_16k=no,speech_angle=float(a),noise_angle=float(b),
                noise_scale=float(scale),gain=float(gain),samples=n,rir=str(out/f'{uid}_rir.npz')))
        (staging/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--catalog',required=True); p.add_argument('--output',required=True,help='New or empty directory; published only after all splits succeed')
    p.add_argument('--count',type=int,default=300); p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--seed',type=int,default=2026); simulate(p.parse_args())
