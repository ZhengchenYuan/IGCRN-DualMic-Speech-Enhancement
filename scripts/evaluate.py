import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from pystoi import stoi
from igcrn.data import Mixtures
from igcrn.model import IGCRN

def sisdr(estimate, target):
    e=estimate.astype(np.float64)-np.mean(estimate); t=target.astype(np.float64)-np.mean(target)
    if np.sum(t*t)<1e-10: raise ValueError('Silent evaluation target')
    projection=np.dot(e,t)*t/np.dot(t,t)
    return float(10*np.log10((np.sum(projection**2)+1e-12)/(np.sum((e-projection)**2)+1e-12)))

def main(a):
    torch.set_num_threads(2)
    ds=Mixtures(a.manifest); models={}; provenance={}
    for mode,path in [('dual',a.dual),('duplicate',a.duplicate)]:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['mode']==mode
        models[mode]=IGCRN(**ck['config']['model']).eval()
        models[mode].load_state_dict(ck['model'])
        provenance[mode]={'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'epoch':ck['epoch']}
        if mode=='dual': reference=ck
        else:
            assert reference['config']==ck['config'] and reference['manifest_hashes']==ck['manifest_hashes']
            assert reference['epoch']==ck['epoch'] and reference['batch_size']==ck['batch_size'], 'Unequal budget'
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); results=[]
    with torch.no_grad():
        for i,r in enumerate(ds.items):
            x,y=ds[i]; target=y.numpy(); base=sisdr(x[0].numpy(),target)
            estimates={'mic1':x[0].numpy()}
            for mode,m in models.items():
                inp=x if mode=='dual' else x[[0,0]]
                estimates[mode]=m(inp[None])[0].numpy()
            for mode,e in estimates.items():
                score=sisdr(e,target)
                results.append(dict(id=r['id'],snr_db=r['snr_db'],method=mode,si_sdr=score,
                                    si_sdri=score-base,stoi=float(stoi(target,e,16000,extended=False))))
            if i<4:
                sf.write(out/f'{r["id"]}_target.wav',target,16000,subtype='FLOAT')
                for mode,e in estimates.items(): sf.write(out/f'{r["id"]}_{mode}.wav',e,16000,subtype='FLOAT')
    with (out/'per_utterance.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    summary=[]
    for snr in ['all']+sorted(set(r['snr_db'] for r in results)):
        for method in estimates:
            selected=[r for r in results if r['method']==method and (snr=='all' or r['snr_db']==snr)]
            summary.append(dict(snr_db=snr,method=method,n=len(selected),
                **{k:float(np.mean([r[k] for r in selected])) for k in ('si_sdr','si_sdri','stoi')}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    provenance['evaluation_manifest_sha256']=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    ranked=sorted([r for r in results if r['method']=='dual'],key=lambda r:r['si_sdri'])
    (out/'listening_candidates.json').write_text(json.dumps({'lowest':ranked[:2],'highest':ranked[-2:]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--dual',required=True)
    p.add_argument('--duplicate',required=True); p.add_argument('--output',required=True); main(p.parse_args())
