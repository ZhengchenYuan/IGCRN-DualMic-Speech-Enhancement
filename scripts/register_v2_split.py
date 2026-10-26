"""Lock all V1 identities before extending a candidate source pool."""
import argparse
from pathlib import Path
from igcrn.data import rows
from igcrn.v2_common import sha,json_write,new_output
from scripts.prepare_data import ordered


def identity(r):
    return str(r['speaker_id'] if r['kind']=='speech' else r['recording_id'])


def register(v1_manifests,catalog,seed,dual_checkpoint=None):
    if len(v1_manifests)!=3: raise ValueError('All three V1 manifests are required; test alone is insufficient')
    roles={'speech':{},'noise':{}}; fingerprints={}; evidence=[]
    def reserve(r,split,fingerprint=None):
        kind=r['kind']; key=identity(r)
        previous=roles[kind].get(key)
        if previous is not None and previous!=split: raise ValueError(f'V1 identity leakage: {kind}/{key}')
        roles[kind][key]=split
        fingerprint=fingerprint or r.get('sha256')
        if fingerprint:
            if fingerprint in fingerprints and fingerprints[fingerprint]!=split:
                raise ValueError('V1 byte-level source leakage')
            fingerprints[fingerprint]=split
    for split,path in zip(('train','valid','test'),v1_manifests):
        entries=rows(path)
        if not entries or any(r['split']!=split for r in entries): raise ValueError(f'Invalid V1 {split} manifest')
        for r in entries:
            reserve(dict(r['speech'],kind='speech'),split,r.get('speech_sha256'))
            reserve(dict(r['noise'],kind='noise'),split,r.get('noise_sha256'))
        evidence.append(dict(split=split,path=str(Path(path).resolve()),sha256=sha(path),count=len(entries)))
    binding=None
    if dual_checkpoint is not None:
        import torch
        ck=torch.load(dual_checkpoint,map_location='cpu',weights_only=False)
        if ck['mode']!='dual' or ck['manifest_hashes']!=[r['sha256'] for r in evidence[:2]]:
            raise ValueError('V1 training/validation manifests do not match the frozen Dual checkpoint')
        binding=dict(sha256=sha(dual_checkpoint),epoch=ck['epoch'])
    candidates=rows(catalog)
    for kind in roles:
        unused={identity(r) for r in candidates if r['kind']==kind}-set(roles[kind])
        for i,key in enumerate(ordered(unused,seed,'v2-'+kind)):
            roles[kind][key]='train' if i%10<8 else ('valid' if i%10==8 else 'test')
    selected=[]
    for r in candidates:
        if not r.get('license') or not r.get('source_url'): raise ValueError('Missing verified source metadata')
        split=roles[r['kind']][identity(r)]
        fingerprint=sha(r['path'])
        if r.get('sha256') and fingerprint!=r['sha256']: raise ValueError('Candidate source changed')
        if fingerprint in fingerprints and fingerprints[fingerprint]!=split:
            raise ValueError('Candidate bytes collide with a protected V1 split')
        fingerprints[fingerprint]=split
        selected.append(dict(r,split=split,sha256=fingerprint))
    for split in ('train','valid','test'):
        for kind in roles:
            if not any(r['split']==split and r['kind']==kind for r in selected):
                raise ValueError(f'No accessible {split}/{kind} sources; restore the proper corpus')
    return dict(schema_version=2,seed=seed,v1_manifests=evidence,candidate_catalog_sha256=sha(catalog),
                roles=roles,catalog=selected,v1_checkpoint=binding)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-manifests',nargs=3,required=True,metavar=('TRAIN','VALID','TEST'))
    p.add_argument('--v1-dual',required=True,help='Trusted V1 checkpoint binding train/valid hashes'); p.add_argument('--catalog',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args(); out=new_output(a.output,[Path(x).parent for x in a.v1_manifests])
    result=register(a.v1_manifests,a.catalog,a.seed,a.v1_dual); json_write(out,result)
    print(out)

if __name__=='__main__': main()
