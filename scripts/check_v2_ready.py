"""Read-only Colab preflight: never install, generate data, or train."""
import argparse
import importlib
import json
import time
from pathlib import Path
import yaml
from igcrn.v2_common import sha
from scripts.simulate_v2 import approved


def pra_version():
    try:
        return importlib.import_module('pyroomacoustics').__version__
    except (ImportError,OSError):
        return None


def check(cfg, registry, physical, diagnostic, dual, data=None, run=None, require_t4=False):
    checks=[]
    def record(name,status,detail): checks.append(dict(check=name,status=status,detail=detail))
    def read(path,name):
        path=Path(path)
        if not path.is_file():
            record(name,'UNVERIFIED','Required file is absent'); return None
        try: return json.loads(path.read_text())
        except (ValueError,OSError) as exc:
            record(name,'FAIL',str(exc)); return None
    try:
        approved(cfg); record('approved_config','PASS','Approved acoustic/data/training settings')
    except (KeyError,ValueError) as exc: record('approved_config','FAIL',str(exc))
    version=pra_version()
    record('pra_runtime','PASS' if version=='0.10.0' else 'UNVERIFIED',version or 'PRA unavailable; no installation attempted')
    report=read(physical,'physical_acceptance')
    if report is not None:
        probes=report.get('full_rt60_probes',[])
        valid=(report.get('status')=='PASS' and {p.get('scene',{}).get('rt60') for p in probes}=={.15,.25,.35}
               and all(p.get('pra_version')==version for p in probes))
        record('physical_acceptance','PASS' if valid else 'UNVERIFIED','Must pass all targets in the installed PRA version')
    reg=read(registry,'split_registry'); evidence=[]
    if reg is not None:
        evidence=reg.get('v1_manifests',[])
        valid=len(evidence)==3 and [r.get('split') for r in evidence]==['train','valid','test']
        if valid:
            for r in evidence:
                p=Path(r['path'])
                if not p.is_file():
                    record('v1_'+r['split']+'_manifest','UNVERIFIED','Original manifest unavailable'); valid=False
                elif sha(p)!=r['sha256']:
                    record('v1_'+r['split']+'_manifest','FAIL','Original manifest hash changed'); valid=False
        record('split_registry','PASS' if valid else 'UNVERIFIED','All THREE original manifests must be available and hash-matched')
        if valid:
            try:
                for item in evidence:
                    entries=[json.loads(line) for line in Path(item['path']).read_text().splitlines() if line.strip()]
                    if not entries: raise ValueError('Empty original V1 manifest')
                    for entry in entries:
                        if entry['split']!=item['split']: raise ValueError('Original V1 split label mismatch')
                        for kind,key in [('speech','speaker_id'),('noise','recording_id')]:
                            ident=str(entry[kind][key])
                            if reg['roles'][kind].get(ident)!=item['split']:
                                raise ValueError('Protected V1 identity was reassigned')
                record('protected_v1_roles','PASS','Every original V1 identity retains its original split')
            except (ValueError,KeyError,TypeError) as exc: record('protected_v1_roles','FAIL',str(exc))
        try:
            if not reg.get('catalog'): raise ValueError('Empty catalog')
            expected={(kind,split) for kind in ('speech','noise') for split in ('train','valid','test')}
            if {(r['kind'],r['split']) for r in reg['catalog']}!=expected: raise ValueError('Missing split/source pool')
            fingerprint_roles={}
            for r in reg['catalog']:
                ident=str(r['speaker_id'] if r['kind']=='speech' else r['recording_id'])
                if reg['roles'][r['kind']][ident]!=r['split']: raise ValueError('Catalog/locked-role mismatch')
                p=Path(r['path'])
                if not p.is_file(): raise FileNotFoundError('Registered raw source unavailable')
                if sha(p)!=r['sha256']: raise ValueError('Registered raw source changed')
                if r['sha256'] in fingerprint_roles and fingerprint_roles[r['sha256']]!=r['split']:
                    raise ValueError('Source bytes cross split boundaries')
                fingerprint_roles[r['sha256']]=r['split']
            record('registered_sources','PASS','Source bytes match locked roles and hashes')
        except FileNotFoundError as exc: record('registered_sources','UNVERIFIED',str(exc))
        except (ValueError,KeyError) as exc: record('registered_sources','FAIL',str(exc))
    checkpoint_hash=None
    if not Path(dual).is_file(): record('v1_checkpoint','UNVERIFIED','Frozen Dual checkpoint unavailable')
    else:
        import torch
        try:
            ck=torch.load(dual,map_location='cpu',weights_only=False)  # Own trusted checkpoint only.
            checkpoint_hash=sha(dual)
            binding=reg.get('v1_checkpoint') if reg else None
            valid=(binding is not None and binding.get('sha256')==checkpoint_hash
                   and binding.get('epoch')==ck['epoch'] and ck['mode']=='dual'
                   and len(evidence)==3 and ck['manifest_hashes']==[r['sha256'] for r in evidence[:2]])
            status='UNVERIFIED' if binding is None or len(evidence)!=3 else 'PASS' if valid else 'FAIL'
            record('v1_checkpoint',status,'Checkpoint must bind the supplied original train/valid manifests')
        except (ValueError,KeyError,OSError) as exc: record('v1_checkpoint','FAIL',str(exc))
    diag=read(diagnostic,'v1_dual_channel_diagnostic')
    if diag is not None:
        valid=(diag.get('status')=='VERIFIED' and diag.get('requested_cases',0)>0
               and diag.get('verified_cases')==diag.get('requested_cases')
               and diag.get('checkpoint_sha256')==checkpoint_hash and len(evidence)==3
               and diag.get('evaluation_manifest_sha256') in {r['sha256'] for r in evidence[1:]})
        record('v1_dual_channel_diagnostic','PASS' if valid else 'UNVERIFIED',
               'Actual validation/test audio diagnostic must match this frozen checkpoint and manifest')
    if data is not None and Path(data).exists():
        try:
            from igcrn.paired import verify_dataset
            info=verify_dataset(data)
            if info['config']!=cfg or info['registry_sha256']!=sha(registry): raise ValueError('Existing V2 dataset provenance changed')
            record('existing_v2_data','PASS','Production manifests/counts/identity isolation verified')
        except (OSError,ValueError,KeyError) as exc: record('existing_v2_data','FAIL',str(exc))
    elif data is not None:
        record('v2_data_action','PASS','WILL_GENERATE after approval; no production dataset is claimed verified')
    if run is not None and (Path(run)/'budget.json').exists():
        budget=read(Path(run)/'budget.json','remaining_budget')
        if budget is not None:
            remaining=budget['deadline_unix']-time.time()
            record('remaining_budget','PASS' if remaining>0 and budget['budget_minutes']==cfg['budget_minutes'] else 'FAIL',
                   dict(remaining_seconds=remaining,budget_reset=False))
    if require_t4:
        import torch
        available=torch.cuda.is_available()
        name=torch.cuda.get_device_name(0) if available else 'CUDA unavailable'
        record('t4_runtime','PASS' if available and 'T4' in name else 'UNVERIFIED',name)
    ready=all(r['status']=='PASS' for r in checks)
    return dict(status='READY_FOR_APPROVED_EXECUTION' if ready else 'BLOCKED',checks=checks,
                scope='inputs_and_t4' if require_t4 else 'inputs_only',
                read_only=True,training_started=False,data_generated=False,
                note='Readiness is not approval. Full training still requires RUN_OVERNIGHT=True / --execute.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/v2.yaml'); p.add_argument('--registry',required=True)
    p.add_argument('--physical',required=True); p.add_argument('--diagnostic',required=True)
    p.add_argument('--v1-dual',required=True); p.add_argument('--data'); p.add_argument('--run')
    p.add_argument('--require-t4',action='store_true'); a=p.parse_args()
    result=check(yaml.safe_load(Path(a.config).read_text()),a.registry,a.physical,a.diagnostic,a.v1_dual,
                 a.data,a.run,a.require_t4)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['status']=='READY_FOR_APPROVED_EXECUTION' else 2)

if __name__=='__main__': main()
