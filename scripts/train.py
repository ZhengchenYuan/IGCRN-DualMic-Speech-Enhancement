import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from igcrn.model import IGCRN, spectral_loss
from igcrn.data import Mixtures

def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False

def main(a):
    cfg=yaml.safe_load(Path(a.config).read_text()); seed_all(cfg['seed']); torch.set_num_threads(2)
    device=torch.device(a.device); model=IGCRN(**cfg['model']).to(device)
    print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'device':str(device)}))
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']); start=0; best=float('inf')
    gen=torch.Generator().manual_seed(cfg['seed'])
    paths=[Path(a.data)/f'{s}.jsonl' for s in ('train','valid')]
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    loaders=[DataLoader(Mixtures(p,a.mode),batch_size=a.batch_size or cfg['batch_size'],
               shuffle=i==0,generator=gen,num_workers=0) for i,p in enumerate(paths)]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if not a.resume and (out/'last.pt').exists():
        raise ValueError('Output already has a checkpoint; resume or use a new directory')
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False) # Trust only your own checkpoints.
        assert ck['config']==cfg and ck['mode']==a.mode and ck['manifest_hashes']==hashes
        assert ck['batch_size']==loaders[0].batch_size
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1; best=ck['best']; gen.set_state(ck['loader_rng'])
        torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
    for epoch in range(start,cfg['epochs']):
        values=[]
        for train,loader in zip((True,False),loaders):
            model.train(train); total=0.; count=0
            with torch.set_grad_enabled(train):
                for x,y in loader:
                    x,y=x.to(device),y.to(device)
                    loss=spectral_loss(model.spectrum(x),model.stft(y))
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
                    if train:
                        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                    total+=loss.item()*len(x); count+=len(x)
            values.append(total/count)
        improved=values[1]<best; best=min(best,values[1])
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,
                config=cfg,mode=a.mode,manifest_hashes=hashes,batch_size=loaders[0].batch_size,
                loader_rng=gen.get_state(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        for name in (['last.pt','best.pt'] if improved else ['last.pt']):
            temporary=out/(name+'.tmp'); torch.save(ck,temporary); temporary.replace(out/name)
        record=dict(epoch=epoch,train_loss=values[0],valid_loss=values[1])
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(record,flush=True)
        if a.stop_after_epoch is not None and epoch >= a.stop_after_epoch: break

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/small.yaml')
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['dual','duplicate'],required=True); p.add_argument('--resume')
    p.add_argument('--stop-after-epoch',type=int,help='Zero-based clean interruption for resume checks'); p.add_argument('--batch-size',type=int); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    main(p.parse_args())
