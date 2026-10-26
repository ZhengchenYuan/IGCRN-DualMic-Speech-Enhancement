"""Approved V2 simulation, CPU only, transactionally published; no V1 writes."""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly,fftconvolve
from scipy.io import wavfile
from math import gcd
from igcrn.acoustics import cached_rirs,scene_for,geometry_index
from igcrn.v2_common import sha,json_write,new_output,rng_for


def approved(cfg):
    expected=dict(sample_rate=16000,seconds=2.,room=[5.,5.,3.],mics=[[2.49,2.5,1.5],[2.51,2.5,1.5]],
                  source_radius=1.5,source_height=1.5,angles_degrees=[-90.,90.],rt60_seconds=[.15,.25,.35],
                  snrs_db=[-3,0,3],backend='pra',ray_tracing=False,randomized_ism=False,max_attempts=20)
    for key,value in expected.items():
        if cfg[key]!=value: raise ValueError(f'Unapproved V2 acoustic setting: {key}')
    if cfg['counts']!={'train':3000,'valid':300,'test':300}: raise ValueError('Production counts must be 3000/300/300')
    if cfg['geometry_counts']!={'train':300,'valid':60,'test':60}: raise ValueError('Unapproved geometry counts')
    if cfg['model']!={'width':64,'depth':6,'n_fft':512,'hop':256} or cfg['lr']!=.0002 or cfg['batch_size']!=1:
        raise ValueError('Unapproved model/loss/training settings')
    if not 1<=cfg['max_epochs']<=11 or not 0<cfg['budget_minutes']<=300:
        raise ValueError('Unapproved epoch/runtime limit')


def segment(path,n,rng):
    x,sr=sf.read(path,always_2d=True,dtype='float64'); x=x.mean(1)
    d=gcd(sr,16000); x=resample_poly(x,16000//d,sr//d)
    if len(x)<n: return None,None
    offset=int(rng.integers(len(x)-n+1)); x=x[offset:offset+n]
    if not np.isfinite(x).all(): return None,None
    return x,offset


def simulate(cfg,registry_path,output,cache,counts=None,deadline=None):
    """counts override is only for explicitly labeled acceptance fixtures."""
    approved(cfg); registry=json.loads(Path(registry_path).read_text())
    if counts is None and not registry.get('v1_checkpoint'):
        raise ValueError('Production registry must be bound to original V1 checkpoint manifest hashes')
    protected=[Path(r['path']).parent for r in registry['v1_manifests']]
    out=new_output(output,protected); out.parent.mkdir(parents=True,exist_ok=True)
    if counts is not None and (set(counts)!={'train','valid','test'} or any(not 1<=n<=9 for n in counts.values())):
        raise ValueError('Acceptance mode is limited to 1-9 samples per split')
    if 'drive' in out.parts: raise ValueError('Raw mixtures belong on local disk, not Drive')
    cache_path=Path(cache).resolve()
    if any(cache_path==p.resolve() or p.resolve() in cache_path.parents for p in protected):
        raise ValueError('Cannot write RIR cache into V1 paths')
    count_map=counts or cfg['counts']; sample_count=round(cfg['seconds']*16000)
    status='acceptance_only' if counts is not None else 'production'
    for r in registry['catalog']:
        if sha(r['path'])!=r['sha256']: raise ValueError('Registered source changed')
    cached_meta={}; start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.'+out.name+'.incomplete-',dir=out.parent) as tmp:
        stage=Path(tmp)
        for split in ('train','valid','test'):
            speech=[r for r in registry['catalog'] if r['split']==split and r['kind']=='speech']
            noise=[r for r in registry['catalog'] if r['split']==split and r['kind']=='noise']
            if not speech or not noise: raise ValueError(f'Missing split sources: {split}')
            entries=[]
            for i in range(count_map[split]):
                if deadline is not None and time.time()>=deadline: raise TimeoutError('Data preparation deadline reached')
                # Acceptance uses a Latin RT60/SNR sweep; production uses independent shuffled geometry cycles.
                index=((i+('train','valid','test').index(split))%cfg['geometry_counts'][split]
                       if counts is not None else geometry_index(cfg,split,i))
                scene=scene_for(cfg,split,index)
                h,meta=cached_rirs(scene,cache,cfg['backend']); cached_meta[meta['cache_key']]=meta
                # Leave a full RIR of source history before the common crop.
                pre=h.shape[-1]-1; crop_start=pre; total=pre+sample_count
                for attempt in range(cfg['max_attempts']):
                    rng=rng_for(cfg['seed'],split,i,attempt)
                    s=speech[int(rng.integers(len(speech)))]; q=noise[int(rng.integers(len(noise)))]
                    clean,so=segment(s['path'],total,rng); ns,no=segment(q['path'],total,rng)
                    if clean is None or ns is None: continue
                    c=np.stack([fftconvolve(clean,h[m,0])[crop_start:crop_start+sample_count] for m in range(2)],1)
                    v=np.stack([fftconvolve(ns,h[m,1])[crop_start:crop_start+sample_count] for m in range(2)],1)
                    ec,en=np.mean(c[:,0]**2),np.mean(v[:,0]**2)
                    if np.isfinite([ec,en]).all() and min(ec,en)>=1e-12: break
                else: raise ValueError(f'V2 unusable source exhaustion: split={split}, sample_index={i}, attempts=20')
                snr=cfg['snrs_db'][i%3]; scale=np.sqrt(ec/en/10**(snr/10))
                x=c+scale*v; gain=.95/max(np.max(np.abs(x)),np.max(np.abs(c)),1e-8)
                uid=f'{split}_{i:05d}'; mix=uid+'_mix.wav'; target=uid+'_target.wav'
                wavfile.write(stage/mix,16000,(x*gain).astype(np.float32))
                wavfile.write(stage/target,16000,(c[:,0]*gain).astype(np.float32))
                entries.append(dict(id=uid,split=split,mixture=str(out/mix),target=str(out/target),
                    mixture_sha256=sha(stage/mix),target_sha256=sha(stage/target),samples=sample_count,snr_db=snr,
                    speech=s,noise=q,speech_sha256=s['sha256'],noise_sha256=q['sha256'],seed=cfg['seed'],
                    speech_offset_16k=so,noise_offset_16k=no,crop_start=crop_start,pre_roll_samples=pre,
                    speech_angle=scene['angles'][0],noise_angle=scene['angles'][1],geometry=scene,
                    rir_key=meta['cache_key'],rir_sha256=meta['sha256'],global_delay_samples=meta['global_delay_samples'],
                    retry_count=attempt,noise_scale=float(scale),gain=float(gain)))
            (stage/f'{split}.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in entries))
        json_write(stage/'dataset.json',dict(schema_version=2,status=status,config=cfg,counts=count_map,
            registry_sha256=sha(registry_path),rir_cache=str(Path(cache).resolve()),rirs=cached_meta,
            manifests={s:sha(stage/f'{s}.jsonl') for s in count_map},seconds=time.monotonic()-start))
        stage.replace(out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/v2.yaml')
    p.add_argument('--registry',required=True); p.add_argument('--output',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--acceptance-count',type=int); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    if a.acceptance_count is not None and not 1<=a.acceptance_count<=9: p.error('Acceptance count must be 1-9')
    counts={s:a.acceptance_count for s in ('train','valid','test')} if a.acceptance_count is not None else None
    print(simulate(cfg,a.registry,a.output,a.cache,counts))

if __name__=='__main__': main()
