import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from types import SimpleNamespace
from scripts.simulate import simulate
from scripts.train import main as train
from scripts.evaluate import main as evaluate

def test_resume_and_evaluation(tmp_path):
    root=tmp_path; catalog=[]
    for j,split in enumerate(('train','valid','test')):
     for k,kind in enumerate(('speech','noise')):
      path=root/f'{split}_{kind}.wav'; t=np.arange(16000)/16000
      signal=(.2*np.sin(2*np.pi*(200+j*20)*t)+.1*np.sin(2*np.pi*600*t)) if k==0 else np.random.default_rng(j).normal(0,.1,len(t))
      sf.write(path,signal,16000,subtype='FLOAT')
      catalog.append(dict(path=str(path),kind=kind,split=split,speaker_id=str(j),recording_id=str(j),license='original diagnostic',source_url='synthetic'))
    p=root/'catalog.jsonl'; p.write_text(''.join(json.dumps(r)+'\n' for r in catalog))
    simulate(SimpleNamespace(catalog=str(p),output=str(root/'data'),seed=2026,count=3,seconds=.5))
    def run(mode,out,resume=None,stop=None):
     train(SimpleNamespace(config='configs/smoke.yaml',device='cpu',data=str(root/'data'),mode=mode,output=str(root/out),batch_size=None,resume=str(resume) if resume else None,stop_after_epoch=stop))
    run('dual','continuous')
    run('dual','resumed',stop=0)
    run('dual','resumed',resume=root/'resumed/last.pt')
    a=torch.load(root/'continuous/last.pt',weights_only=False); b=torch.load(root/'resumed/last.pt',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model']), 'Resume mismatch'
    print('Resume matches uninterrupted model bitwise')
    run('duplicate','duplicate')
    evaluate(SimpleNamespace(manifest=str(root/'data/test.jsonl'),dual=str(root/'continuous/last.pt'),duplicate=str(root/'duplicate/last.pt'),output=str(root/'evaluation')))
    assert len(json.loads((root/'evaluation/summary.json').read_text()))==12
    print('End-to-end diagnostic evaluation passed; no research results:',root)
