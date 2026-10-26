"""Read-only V1 inputs; write diagnostics to a new output directory only."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.fft import next_fast_len
from igcrn.v2_common import sha,json_write,new_output
from igcrn.data import rows


def signal_stats(x,fs=16000):
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all(): raise ValueError('Expected finite N,2 synchronized audio')
    power=np.mean(x*x,axis=0)
    if np.min(power)<1e-12: raise ValueError('Silent channel')
    nfft=next_fast_len(2*len(x)-1); interpolation=32; width=2*interpolation
    cross=np.fft.rfft(x[:,1],nfft)*np.conj(np.fft.rfft(x[:,0],nfft))
    cc=np.fft.irfft(cross/np.maximum(np.abs(cross),1e-12),nfft*interpolation)
    window=np.r_[cc[-width:],cc[:width+1]]
    lag=(int(np.argmax(window))-width)/interpolation
    f,_,z1=stft(x[:,0],fs=fs,nperseg=512,noverlap=256,boundary='zeros')
    _,_,z2=stft(x[:,1],fs=fs,nperseg=512,noverlap=256,boundary='zeros')
    weight=np.abs(z1)*np.abs(z2); active=weight>max(float(weight.max())*1e-6,1e-15)
    ipd=np.angle(z2*np.conj(z1)); ild=20*np.log10((np.abs(z2)+1e-12)/(np.abs(z1)+1e-12)); bands={}
    for low,high in [(200,1000),(1000,4000),(4000,7000)]:
        mask=active & ((f>=low)&(f<high))[:,None]; w=weight[mask]
        if not len(w): continue
        circular=np.sum(w*np.exp(1j*ipd[mask]))/w.sum()
        bands[f'{low}-{high}Hz']=dict(ipd_circular_mean_rad=float(np.angle(circular)),
            ipd_concentration=float(abs(circular)),weighted_abs_ipd_rad=float(np.average(abs(ipd[mask]),weights=w)),
            ild_mean_db=float(np.average(ild[mask],weights=w)))
    return dict(correlation=float(np.corrcoef(x.T)[0,1]),difference_energy_ratio=float(np.mean((x[:,1]-x[:,0])**2)/power[0]),
                gcc_phat_tdoa_samples=lag,gcc_search_limit_samples=2.,bands=bands)


def theoretical_delay(angle):
    rad=np.deg2rad(angle); source=np.array([2.5+1.5*np.sin(rad),2.5+1.5*np.cos(rad),1.5])
    m1=np.array([2.49,2.5,1.5]); m2=np.array([2.51,2.5,1.5])
    return float((np.linalg.norm(source-m2)-np.linalg.norm(source-m1))/343*16000)


def diagnose(manifest,checkpoint,output,audio_root=None,limit=120):
    import torch
    from igcrn.model import IGCRN
    from scripts.evaluate import sisdr
    from pystoi import stoi
    out=new_output(output,[Path(manifest).parent,Path(checkpoint).parent]); out.mkdir(parents=True)
    torch.set_num_threads(2)
    checkpoint_hash=sha(checkpoint); manifest_hash=sha(manifest)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['mode']!='dual': raise ValueError('Need frozen trained Dual checkpoint')
    model=IGCRN(**ck['config']['model']).eval(); model.load_state_dict(ck['model'])
    entries=rows(manifest)[:limit]; cases=[]; missing=[]
    predictions=[]
    for r in entries:
        if 'speech_angle' in r and 'noise_angle' in r:
            ts,tn=theoretical_delay(r['speech_angle']),theoretical_delay(r['noise_angle'])
            predictions.append(dict(id=r['id'],speech_angle=r['speech_angle'],noise_angle=r['noise_angle'],
                speech_tdoa_samples=ts,noise_tdoa_samples=tn,absolute_tdoa_separation_samples=abs(ts-tn)))
    for r in entries:
        mixture=Path(r['mixture']); target=Path(r['target'])
        if audio_root:
            mixture=Path(audio_root)/mixture.name; target=Path(audio_root)/target.name
        if not mixture.is_file() or not target.is_file():
            missing.append(dict(id=r['id'],mixture_available=mixture.is_file(),target_available=target.is_file())); continue
        if sha(mixture)!=r['mixture_sha256'] or sha(target)!=r['target_sha256']: raise ValueError('V1 frozen audio hash mismatch')
        x,sr=sf.read(mixture,always_2d=True,dtype='float32'); y,sy=sf.read(target,dtype='float32')
        if sr!=sy or sr!=16000 or x.shape!=(len(y),2): raise ValueError('V1 shape/rate mismatch')
        result=dict(id=r['id'],snr_db=r['snr_db'],statistics=signal_stats(x),conditions={})
        if 'speech_angle' in r:
            ts=theoretical_delay(r['speech_angle']); tn=theoretical_delay(r['noise_angle'])
            result['geometry_prediction']=dict(speech_tdoa_samples=ts,noise_tdoa_samples=tn,
                absolute_tdoa_separation_samples=abs(ts-tn),note='Geometry prediction, not measured source-specific TDOA')
        base=sisdr(x[:,0],y)
        conditions={'normal':x,'duplicated_mic1':x[:,[0,0]],'zero_mic2':np.stack([x[:,0],np.zeros(len(x),np.float32)],1)}
        with torch.no_grad():
            normal=None
            for label,value in conditions.items():
                estimate=model(torch.from_numpy(value.T.copy())[None])[0].numpy()
                if not np.isfinite(estimate).all(): raise ValueError('Non-finite frozen prediction')
                if normal is None: normal=estimate.copy()
                result['conditions'][label]=dict(si_sdri=sisdr(estimate,y)-base,stoi=float(stoi(y,estimate,16000,extended=False)),
                    output_difference_ratio=float(np.sum((estimate-normal)**2)/max(np.sum(normal**2),1e-12)))
        cases.append(result)
    if sha(checkpoint)!=checkpoint_hash or sha(manifest)!=manifest_hash: raise ValueError('V1 input was modified')
    geometry_summary=None
    if predictions:
        gaps=np.array([r['absolute_tdoa_separation_samples'] for r in predictions])
        geometry_summary=dict(source='manifest angles with documented V1 geometry, not audio measurements',
            cases=len(predictions),separation_samples_quantiles=np.quantile(gaps,[0,.25,.5,.75,1]).tolist(),
            below_0_1_samples=int(np.sum(gaps<.1)),below_0_25_samples=int(np.sum(gaps<.25)))
    report=dict(geometry_summary=geometry_summary,geometry_predictions=predictions,status='VERIFIED' if len(cases)==len(entries) and cases else 'UNVERIFIED' if not cases else 'PARTIAL',
        checkpoint_sha256=checkpoint_hash,evaluation_manifest_sha256=manifest_hash,epoch=ck['epoch'],config=ck['config'],
        manifest_hashes=ck['manifest_hashes'],requested_cases=len(entries),verified_cases=len(cases),missing=missing,cases=cases,
        limitations=['Zero/duplicated Mic2 are diagnostic perturbations, not the independently trained ablation.',
                     'GCC-PHAT on a mixture is not the isolated speech/noise propagation delay.',
                     'A high inter-channel correlation alone does not imply no usable spatial cues.',
                     'If this is a test manifest these are retrospective diagnostics, not tuning data.'])
    json_write(out/'diagnostic.json',report); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True)
    p.add_argument('--dual',required=True); p.add_argument('--output',required=True); p.add_argument('--audio-root')
    p.add_argument('--limit',type=int,default=120); a=p.parse_args()
    print(diagnose(a.manifest,a.dual,a.output,a.audio_root,a.limit)['status'])

if __name__=='__main__': main()
