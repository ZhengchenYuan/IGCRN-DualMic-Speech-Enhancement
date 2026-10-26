"""Small actual-PRA fixture acceptance. No training, no public-data download."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import fftconvolve
from igcrn.v2_common import new_output,json_write,sha
from scripts.register_v2_split import register
from scripts.simulate_v2 import simulate


def acceptance(output):
    root=new_output(output); root.mkdir(parents=True); cfg=yaml.safe_load(Path('configs/v2.yaml').read_text())
    catalog=[]; manifests=[]; v1=root/'synthetic_v1_fixture'; v1.mkdir(); started=time.monotonic()
    for i,split in enumerate(('train','valid','test')):
        entries=[]
        for j,kind in enumerate(('speech','noise')):
            p=root/f'original_fixture_{split}_{kind}.wav'
            sf.write(p,np.random.default_rng(2026+i*2+j).normal(0,.1,64000),16000,subtype='PCM_16')
            r=dict(path=str(p),kind=kind,split=split,speaker_id=f'fixture-{i}',recording_id=f'fixture-{i}',
                license='CC0 original synthetic acceptance signal',source_url='synthetic://seed/2026',sha256=sha(p))
            catalog.append(r); entries.append(r)
        manifest=v1/f'{split}.jsonl'; manifest.write_text(json.dumps(dict(split=split,speech=entries[0],noise=entries[1]))+'\n')
        manifests.append(manifest)
    cat=root/'catalog.jsonl'; cat.write_text(''.join(json.dumps(r)+'\n' for r in catalog))
    registry=root/'registry.json'; json_write(registry,register(manifests,cat,2026))
    counts=dict(train=3,valid=3,test=3)
    simulate(cfg,registry,root/'first',root/'cache1',counts=counts)
    simulate(cfg,registry,root/'second',root/'cache2',counts=counts)
    snr_errors=[]; target_errors=[]; identical=True; rt60_seen=set()
    for split in counts:
        a=[json.loads(x) for x in (root/'first'/f'{split}.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (root/'second'/f'{split}.jsonl').read_text().splitlines()]
        for r,s in zip(a,b):
            identical &= all(r[k]==s[k] for k in ('mixture_sha256','target_sha256','rir_sha256'))
            rt60_seen.add(r['geometry']['rt60'])
            x,_=sf.read(r['mixture']); y,_=sf.read(r['target'])
            snr_errors.append(abs(10*np.log10(np.mean(y*y)/np.mean((x[:,0]-y)**2))-r['snr_db']))
            with np.load(root/'cache1'/(r['rir_key']+'.npz')) as f:h=f['rir']
            source,_=sf.read(r['speech']['path']); start=r['speech_offset_16k']; pre=r['crop_start']
            expected=fftconvolve(source[start:start+pre+32000],h[0,0])[pre:pre+32000]*r['gain']
            target_errors.append(float(np.max(abs(y-expected))))
    passed=identical and max(snr_errors)<1e-5 and max(target_errors)<1e-7 and rt60_seen=={.15,.25,.35}
    report=dict(status='PASS' if passed else 'FAIL',source='generated Gaussian fixtures, NOT real speech or V1 waveforms',
        training_started=False,counts_per_generation=counts,independent_generations=2,all_rt60_targets=sorted(rt60_seen),
        identical_audio_and_rir_hashes=bool(identical),max_snr_error_db=max(snr_errors),
        max_clean_mic1_target_error=max(target_errors),seconds=time.monotonic()-started,
        limitations=['Actual V1 identity registry and real-source generation require all original manifests and source files.'])
    json_write(root/'acceptance.json',report)
    if not passed: raise ValueError(report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(acceptance(p.parse_args().output),indent=2))
