import json
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from scripts.simulate import simulate
from igcrn.data import Mixtures, rows

def test_reproducible_simulation_and_snr(tmp_path):
    catalog=[]
    for j,split in enumerate(('train','valid','test')):
        for k,kind in enumerate(('speech','noise')):
            p=tmp_path/f'{split}_{kind}.wav'
            sf.write(p,np.random.default_rng(j*2+k).normal(0,.1,16000),16000,subtype='FLOAT')
            catalog.append(dict(path=str(p),kind=kind,split=split,speaker_id=str(j),recording_id=str(j),license='original test signal',source_url='synthetic'))
    path=tmp_path/'catalog.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in catalog))
    a=SimpleNamespace(catalog=str(path),output=str(tmp_path/'a'),seed=2026,count=3,seconds=.5)
    simulate(a); a.output=str(tmp_path/'b'); simulate(a)
    ra=rows(tmp_path/'a/test.jsonl'); rb=rows(tmp_path/'b/test.jsonl')
    assert [r['mixture_sha256'] for r in ra]==[r['mixture_sha256'] for r in rb]
    for r in ra:
        x,_=sf.read(r['mixture']); y,_=sf.read(r['target'])
        measured=10*np.log10(np.mean(y*y)/np.mean((x[:,0]-y)**2))
        assert abs(measured-r['snr_db'])<1e-5
    ds=Mixtures(tmp_path/'a/test.jsonl','duplicate'); x,y=ds[0]
    assert np.array_equal(x[0],x[1])
