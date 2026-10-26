"""Complete paired evaluation under the shared deadline. Metrics unchanged."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from pystoi import stoi
from igcrn.model import IGCRN
from igcrn.data import Mixtures
from igcrn.v2_common import sha,json_write
from scripts.evaluate import sisdr


def evaluate(data,run,deadline=None,allow_acceptance=False):
    from igcrn.paired import verify_dataset
    info=verify_dataset(data,allow_acceptance)
    run=Path(run); pair=json.loads((run/'pair.json').read_text()); ds=Mixtures(Path(data)/'test.jsonl')
    if not allow_acceptance and len(ds)!=300: raise ValueError('Full V2 test set must contain 300 cases')
    if pair['manifest_hashes']!=[info['manifests'][s] for s in ('train','valid')]: raise ValueError('Evaluation dataset provenance mismatch')
    models={}
    for mode in ('dual','duplicate'):
        record=pair['checkpoints'][mode]
        if sha(record['path'])!=record['sha256']: raise ValueError('Paired checkpoint hash mismatch')
        ck=torch.load(record['path'],map_location='cpu',weights_only=False)
        if (ck['epoch']+1!=pair['completed_epochs'] or ck['global_step']!=pair['global_step']
                or ck['mode']!=mode or ck['config']!=pair['config']): raise ValueError('Unequal evaluation budget')
        model=IGCRN(**ck['config']['model']).eval(); model.load_state_dict(ck['model']); models[mode]=model
    out=run/'evaluation'; out.mkdir(exist_ok=True); results=[]; times=[]; started=time.monotonic()
    with torch.no_grad():
        for i,r in enumerate(ds.items):
            estimate=max([4.]+times[-10:])*1.25
            if deadline is not None and time.time()+estimate+60>=deadline:
                json_write(out/'partial.json',dict(status='EVALUATION_INCOMPLETE',rows=results))
                return False
            tick=time.monotonic(); x,y=ds[i]; target=y.numpy(); base=sisdr(x[0].numpy(),target)
            estimates={'mic1':x[0].numpy()}
            for mode,model in models.items(): estimates[mode]=model((x if mode=='dual' else x[[0,0]])[None])[0].numpy()
            for mode,value in estimates.items():
                if not np.isfinite(value).all(): raise ValueError('Non-finite prediction')
                score=sisdr(value,target); intelligibility=float(stoi(target,value,16000,extended=False))
                if not np.isfinite([score,intelligibility]).all(): raise ValueError('Non-finite metric')
                results.append(dict(id=r['id'],snr_db=r['snr_db'],method=mode,si_sdr=score,
                                    si_sdri=score-base,stoi=intelligibility))
            if i<4:
                wavfile.write(out/f'{r["id"]}_input.wav',16000,x.T.numpy())
                wavfile.write(out/f'{r["id"]}_target.wav',16000,target)
                for mode,value in estimates.items(): wavfile.write(out/f'{r["id"]}_{mode}.wav',16000,value.astype(np.float32))
            times.append(time.monotonic()-tick)
    if len(results)!=3*len(ds): raise ValueError('Incomplete metrics')
    with (out/'per_utterance.csv.tmp').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (out/'per_utterance.csv.tmp').replace(out/'per_utterance.csv')
    summary=[]
    for snr in ['all']+sorted({r['snr_db'] for r in results}):
        for mode in ('mic1','dual','duplicate'):
            chosen=[r for r in results if r['method']==mode and (snr=='all' or r['snr_db']==snr)]
            summary.append(dict(method=mode,snr_db=snr,n=len(chosen),
                **{k:float(np.mean([r[k] for r in chosen])) for k in ('si_sdr','si_sdri','stoi')}))
    json_write(out/'summary.json',summary)
    by_id={r['id']:{mode:next(t for t in results if t['id']==r['id'] and t['method']==mode)
                   for mode in ('dual','duplicate')} for r in ds.items}
    rng=np.random.default_rng(2026); ci={}
    for metric in ('si_sdri','stoi'):
        deltas=np.array([by_id[r['id']]['dual'][metric]-by_id[r['id']]['duplicate'][metric] for r in ds.items])
        strata=[np.array([i for i,r in enumerate(ds.items) if r['snr_db']==snr]) for snr in sorted({r['snr_db'] for r in ds.items})]
        samples=[np.mean(deltas[np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])]) for _ in range(2000)]
        ci[metric]=dict(dual_minus_duplicate=float(deltas.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
    json_write(out/'paired_ci.json',dict(seed=2026,replicates=2000,method='utterance-paired SNR-stratified bootstrap',
        caveat='Not a speaker/noise-cluster or multi-seed confidence interval',metrics=ci))
    ranked=sorted([r for r in results if r['method']=='dual'],key=lambda r:r['si_sdri'])
    json_write(out/'listening_candidates.json',dict(lowest=ranked[:2],highest=ranked[-2:],listened=False))
    json_write(out/'provenance.json',dict(evaluation_manifest_sha256=sha(Path(data)/'test.jsonl'),pair=pair,
        seconds=time.monotonic()-started,case_count=len(ds),metric_rows=len(results),device='cpu'))
    if (out/'partial.json').exists(): (out/'partial.json').replace(out/'partial_before_complete.json')
    return True
