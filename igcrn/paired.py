"""Epoch-boundary paired training; no changes to IGCRN or its objective."""
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Mixtures
from .model import IGCRN,spectral_loss
from .v2_common import sha,json_write,start_budget
from scripts.train import seed_all


class EpochInterrupted(RuntimeError): pass


def rng_state():
    return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python']); np.random.set_state(state['numpy'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def verify_dataset(data,allow_acceptance=False):
    data=Path(data); info=json.loads((data/'dataset.json').read_text())
    if info['status']!='production' and not allow_acceptance: raise ValueError('Acceptance fixtures cannot be trained as production')
    if not allow_acceptance and info['counts']!={'train':3000,'valid':300,'test':300}: raise ValueError('Wrong production data size')
    for split in ('train','valid','test'):
        if sha(data/f'{split}.jsonl')!=info['manifests'][split]: raise ValueError('Frozen manifest changed')
        ds=Mixtures(data/f'{split}.jsonl')
        if len(ds)!=info['counts'][split]: raise ValueError('Manifest count mismatch')
        if any(r['split']!=split for r in ds.items): raise ValueError('Wrong split labels')
    groups={'speech':{},'noise':{}}
    from .data import rows
    for split in ('train','valid','test'):
        for r in rows(data/f'{split}.jsonl'):
            for kind,key in [('speech','speaker_id'),('noise','recording_id')]:
                ident=str(r[kind][key]); old=groups[kind].get(ident)
                if old is not None and old!=split: raise ValueError('Dataset identity leakage')
                groups[kind][ident]=split
    return info


def create_arm(cfg,data,mode,device):
    seed_all(cfg['seed'])
    model=IGCRN(**cfg['model']).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    loaders=[DataLoader(Mixtures(Path(data)/f'{split}.jsonl',mode),batch_size=cfg['batch_size'],
             shuffle=split=='train',generator=generator,num_workers=0) for split in ('train','valid')]
    return dict(model=model,optimizer=optimizer,generator=generator,loaders=loaders,rng=rng_state(),steps=0)


def one_epoch(arm,device,deadline):
    """Interrupt only between updates; incomplete epoch is NOT resumable/committed."""
    restore_rng(arm['rng']); losses=[]; start=time.monotonic(); last_monitor=start
    for train,loader in zip((True,False),arm['loaders']):
        arm['model'].train(train); total=0.; count=0
        with torch.set_grad_enabled(train):
            for x,y in loader:
                if time.time()>=deadline: raise EpochInterrupted('Deadline reached between updates; rollback to last paired epoch')
                x,y=x.to(device),y.to(device)
                loss=spectral_loss(arm['model'].spectrum(x),arm['model'].stft(y))
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                if train:
                    arm['optimizer'].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(arm['model'].parameters(),5)
                    arm['optimizer'].step()
                    if device.type=='cuda': torch.cuda.synchronize()
                    arm['steps']+=1
                total+=float(loss.detach())*len(x); count+=len(x)
                if arm.get('monitor') and time.monotonic()-last_monitor>=60:
                    arm['monitor'](dict(phase='train' if train else 'valid',observed_steps=arm['steps'],loss=total/count))
                    last_monitor=time.monotonic()
        losses.append(total/count)
    arm['rng']=rng_state()
    return dict(train_loss=losses[0],valid_loss=losses[1],seconds=time.monotonic()-start)


def checkpoint(arm,cfg,mode,epoch,hashes):
    return dict(model=arm['model'].state_dict(),optimizer=arm['optimizer'].state_dict(),config=cfg,mode=mode,
                epoch=epoch,global_step=arm['steps'],batch_size=cfg['batch_size'],manifest_hashes=hashes,
                loader_rng=arm['generator'].get_state(),rng=arm['rng'],resume_policy='complete_epoch_only')


def commit_pair(run,arms,cfg,epoch,hashes,logs):
    """Immutable files first; pair.json is the sole publication commit point.

    On mounted Drive, hash re-read verifies transferred bytes; do not assume
    rename is a distributed transaction. An old pair.json stays authoritative.
    """
    run=Path(run); directory=run/f'epoch_{epoch+1:03d}'; directory.mkdir(exist_ok=True)
    files={}
    import tempfile
    with tempfile.TemporaryDirectory(prefix='igcrn-v2-save-') as tmp:
        for mode,arm in arms.items():
            local=Path(tmp)/(mode+'.pt'); torch.save(checkpoint(arm,cfg,mode,epoch,hashes),local)
            target=directory/(mode+'.pt'); expected=sha(local)
            if target.exists() and sha(target)!=expected:
                # Uncommitted/orphan files may exist after a crash, never overwrite a published pair.
                published=json.loads((run/'pair.json').read_text()) if (run/'pair.json').exists() else None
                if published and published['completed_epochs']>=epoch+1: raise ValueError('Refuse to overwrite published checkpoint')
            partial=target.with_suffix('.pt.partial'); shutil.copyfile(local,partial)
            if sha(partial)!=expected: raise IOError('Drive checkpoint verification failed')
            partial.replace(target); files[mode]=dict(path=str(target.resolve()),sha256=expected)
    json_write(directory/'logs.json',logs)
    expected=(epoch+1)*len(arms['dual']['loaders'][0])
    if any(arm['steps']!=expected for arm in arms.values()): raise ValueError('Incomplete or unequal updates')
    record=dict(schema_version=2,completed_epochs=epoch+1,global_step=arms['dual']['steps'],checkpoints=files,
                config=cfg,manifest_hashes=hashes)
    json_write(run/'pair.json',record)
    return record


def restore_pair(run,arms,cfg,hashes,device):
    record=json.loads((Path(run)/'pair.json').read_text())
    if record['config']!=cfg or record['manifest_hashes']!=hashes: raise ValueError('Resume config/data mismatch')
    for mode,arm in arms.items():
        file=record['checkpoints'][mode]
        if sha(file['path'])!=file['sha256']: raise ValueError('Checkpoint corrupted')
        ck=torch.load(file['path'],map_location='cpu',weights_only=False)
        if ck['mode']!=mode or ck['global_step']!=record['global_step']: raise ValueError('Checkpoint provenance mismatch')
        arm['model'].load_state_dict(ck['model']); arm['optimizer'].load_state_dict(ck['optimizer'])
        arm['generator'].set_state(ck['loader_rng']); arm['rng']=ck['rng']; arm['steps']=ck['global_step']
    return record['completed_epochs']


def run_pairs(cfg,data,run,device='cuda',allow_acceptance=False,max_pairs_this_call=None):
    """Both models remain resident; switching epochs does not reload optimizer state."""
    run=Path(run); run.mkdir(parents=True,exist_ok=True)
    owner=run/'owner.json'
    if not owner.exists():
        if any(run.iterdir()): raise ValueError('Not an owned V2 run directory')
        json_write(owner,dict(schema_version=2,config=cfg,data=str(Path(data).resolve())))
    saved=json.loads(owner.read_text())
    if saved!=dict(schema_version=2,config=cfg,data=str(Path(data).resolve())): raise ValueError('Run ownership/config mismatch')
    if not allow_acceptance:
        if cfg['model']!={'width':64,'depth':6,'n_fft':512,'hop':256} or cfg['lr']!=.0002 or cfg['batch_size']!=1:
            raise ValueError('Unapproved model/training changes')
        if cfg['max_epochs']>11 or cfg['budget_minutes']>300: raise ValueError('Unapproved runtime budget')
        if device!='cuda' or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
            raise ValueError('Production execution requires the approved T4 GPU')
    budget=start_budget(run/'budget.json',cfg['budget_minutes'])
    if time.time()>=budget['deadline_unix']:
        json_write(run/'state.json',dict(status='STOPPED_EXPIRED_BUDGET',completed_epochs=json.loads((run/'pair.json').read_text())['completed_epochs'] if (run/'pair.json').exists() else 0))
        return 'STOPPED_EXPIRED_BUDGET'
    info=verify_dataset(data,allow_acceptance)
    if allow_acceptance and (info['status']!='acceptance_only' or max(info['counts'].values())>9
                            or cfg['max_epochs']>2 or cfg['model']['width']>8 or cfg['model']['depth']>2):
        raise ValueError('Local acceptance optimizer tests must stay tiny; no production training bypass')
    hashes=[info['manifests'][s] for s in ('train','valid')]
    device=torch.device(device); torch.set_num_threads(2)
    arms={mode:create_arm(cfg,data,mode,device) for mode in ('dual','duplicate')}
    completed=restore_pair(run,arms,cfg,hashes,device) if (run/'pair.json').exists() else 0
    histories=json.loads((run/'history.json').read_text()) if (run/'history.json').exists() else []
    # A checkpoint transaction may commit just before history publication.
    histories=histories[:completed]
    for e in range(len(histories),completed): histories.append(json.loads((run/f'epoch_{e+1:03d}'/'logs.json').read_text()))
    reserve=60*(cfg['evaluation_reserve_minutes']+cfg['sync_reserve_minutes']+cfg['safety_minutes'])
    estimate=cfg['initial_pair_estimate_minutes']*60; pairs=0; status='TRAINING'
    for epoch in range(completed,cfg['max_epochs']):
        estimate=max(r['pair_seconds']*1.15 for r in histories[-3:]) if histories else estimate
        remaining=budget['deadline_unix']-time.time()
        if remaining<reserve+estimate:
            status='PAIRS_FINISHED_TIME_BUDGET' if completed else 'STOPPED_NO_PAIR'; break
        logs=dict(epoch=epoch,started_unix=time.time()); pair_start=time.monotonic()
        try:
            for mode in ('dual','duplicate'):
                json_write(run/'state.json',dict(status='TRAINING_'+mode.upper(),completed_epochs=completed,
                    active_epoch=epoch,remaining_seconds=budget['deadline_unix']-time.time()))
                def monitor(info,mode=mode):
                    json_write(run/'state.json',dict(status='TRAINING_'+mode.upper(),completed_epochs=completed,
                        active_epoch=epoch,remaining_seconds=budget['deadline_unix']-time.time(),**info))
                arms[mode]['monitor']=monitor
                logs[mode]=one_epoch(arms[mode],device,budget['deadline_unix']-reserve)
            logs['pair_seconds']=time.monotonic()-pair_start
            commit_pair(run,arms,cfg,epoch,hashes,logs)
            logs['pair_seconds']=time.monotonic()-pair_start
            json_write(run/f'epoch_{epoch+1:03d}'/'logs.json',logs)
        except (EpochInterrupted,KeyboardInterrupt):
            json_write(run/'state.json',dict(status='INTERRUPTED_UNCOMMITTED_PAIR',completed_epochs=completed,
                attempted_steps={m:a['steps'] for m,a in arms.items()},resume_policy='rollback_to_last_complete_pair'))
            return 'INTERRUPTED_UNCOMMITTED_PAIR'
        completed=epoch+1; histories.append(logs); json_write(run/'history.json',histories); pairs+=1
        if max_pairs_this_call and pairs>=max_pairs_this_call:
            status='PAUSED_BY_CALLER'; break
    else: status='PAIRS_FINISHED_MAX_EPOCHS'
    json_write(run/'state.json',dict(status=status,completed_epochs=completed,
        equal_committed_updates=True,remaining_seconds=budget['deadline_unix']-time.time(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None))
    return status
