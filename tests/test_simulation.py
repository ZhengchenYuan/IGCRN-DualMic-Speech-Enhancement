import json
from types import SimpleNamespace
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf
from scripts.simulate import simulate
from scripts import simulate as simulator
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


def source_catalog(tmp_path, bad_kind=None, bad_level=0., all_bad=False):
    """Original test audio: unique bytes even for silent files in different splits."""
    catalog = []
    for j, split in enumerate(('train', 'valid', 'test')):
        for k, kind in enumerate(('speech', 'noise')):
            for bad in ([True] if all_bad and kind == bad_kind else
                        [False, True] if kind == bad_kind else [False]):
                path = tmp_path/f'{split}_{kind}_{bad}.wav'
                size = 4096+j*101+k*29
                data = (np.full(size, bad_level) if bad else
                        np.random.default_rng(j*10+k).normal(0,.1,size))
                sf.write(path, data, 16000, subtype='FLOAT')
                catalog.append(dict(path=str(path),kind=kind,split=split,
                    speaker_id=f'speaker-{j}',recording_id=f'noise-{j}',
                    license='original test signal',source_url='synthetic'))
    path=tmp_path/'catalog.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in catalog))
    return path


def simulation_args(tmp_path, catalog, count=9):
    return SimpleNamespace(catalog=str(catalog),output=str(tmp_path/'mixtures'),seed=2026,
                           count=count,seconds=.032)


@pytest.mark.parametrize('kind,level',[('speech',0.),('noise',0.),('speech',1e-8),('noise',1e-8)])
def test_silent_and_near_silent_recovery_is_deterministic(tmp_path,kind,level):
    catalog=source_catalog(tmp_path,bad_kind=kind,bad_level=level)
    args=simulation_args(tmp_path,catalog,count=12)
    simulate(args)
    other=tmp_path/'repeat'; args.output=str(other); simulate(args)
    retries=[]
    for split,count in [('train',12),('valid',3),('test',3)]:
        left=rows(tmp_path/f'mixtures/{split}.jsonl'); right=rows(other/f'{split}.jsonl')
        assert len(left)==len(right)==count
        for i,(a,b) in enumerate(zip(left,right)):
            retries.append(a['retry_count'])
            assert 0 <= a['retry_count'] < 20 and a['snr_db']==[-3,0,3][i%3]
            assert a['speech']['split']==a['noise']['split']==split
            assert '_False.wav' in a[kind]['path']
            # Compare the full metadata after relocating only generated-output paths.
            for key in ('mixture','target','rir'):
                b[key]=str(tmp_path/'mixtures'/Path(b[key]).name)
            assert a==b
            x,_=sf.read(a['mixture']); target,_=sf.read(a['target'])
            measured=10*np.log10(np.mean(target**2)/np.mean((x[:,0]-target)**2))
            assert abs(measured-a['snr_db'])<1e-5
        x,y=Mixtures(tmp_path/f'mixtures/{split}.jsonl')[0]
        assert x.shape==(2,512) and y.shape==(512,)
    assert max(retries)>0
    assert not list(tmp_path.glob('.*.incomplete-*'))


@pytest.mark.parametrize('kind',['speech','noise'])
def test_exhaustion_is_bounded_and_publishes_nothing(tmp_path,monkeypatch,kind):
    args=simulation_args(tmp_path,source_catalog(tmp_path,bad_kind=kind,all_bad=True))
    calls=[]; original=simulator.read
    def counting_read(path,n,rng):
        calls.append(path)
        return original(path,n,rng)
    monkeypatch.setattr(simulator,'read',counting_read)
    with pytest.raises(ValueError,match=r'split=train, sample_index=0 .*attempts=20'):
        simulate(args)
    assert len(calls)==40  # Both real source segments were read on exactly 20 attempts.
    assert not Path(args.output).exists()
    assert not list(tmp_path.glob('.*.incomplete-*'))


def test_last_attempt_can_succeed_and_records_nineteen_retries(tmp_path,monkeypatch):
    args=simulation_args(tmp_path,source_catalog(tmp_path))
    original=simulator.read; calls=0
    def first_nineteen_silent(path,n,rng):
        nonlocal calls
        calls+=1
        audio,offset=original(path,n,rng)
        if calls<=38: audio=audio*0
        return audio,offset
    monkeypatch.setattr(simulator,'read',first_nineteen_silent)
    simulate(args)
    entries=rows(Path(args.output)/'train.jsonl')
    assert entries[0]['retry_count']==19
    assert all(r['retry_count']==0 for r in entries[1:])


@pytest.mark.parametrize('error',[KeyboardInterrupt,OSError])
def test_interruption_after_completed_train_split_leaves_no_manifests(tmp_path,monkeypatch,error):
    args=simulation_args(tmp_path,source_catalog(tmp_path),count=3)
    original=simulator.read
    def fail_in_validation(path,n,rng):
        if 'valid_' in path: raise error('test interruption')
        return original(path,n,rng)
    monkeypatch.setattr(simulator,'read',fail_in_validation)
    with pytest.raises(error): simulate(args)
    assert not Path(args.output).exists()
    assert not list(tmp_path.glob('.*.incomplete-*'))


def test_six_hundred_sample_budget_and_snr_schedule(tmp_path):
    args=simulation_args(tmp_path,source_catalog(tmp_path),count=600)
    # Existing empty output directories are also valid publication targets.
    Path(args.output).mkdir()
    simulate(args)
    for split,count in [('train',600),('valid',120),('test',120)]:
        entries=rows(Path(args.output)/f'{split}.jsonl')
        assert len(entries)==count
        assert [r['id'] for r in entries]==[f'{split}_{i:05d}' for i in range(count)]
        assert [r['snr_db'] for r in entries]==([-3,0,3]*(count//3))
        assert all(r['retry_count']==0 for r in entries)
        assert all(Path(r['mixture']).is_file() and Path(r['target']).is_file() and Path(r['rir']).is_file()
                   for r in entries)
    preserved=(Path(args.output)/'train.jsonl').read_bytes()
    with pytest.raises(FileExistsError,match='new or empty'): simulate(args)
    assert (Path(args.output)/'train.jsonl').read_bytes()==preserved


def test_late_retry_exhaustion_identifies_sample_and_discards_earlier_splits(tmp_path,monkeypatch):
    args=simulation_args(tmp_path,source_catalog(tmp_path),count=3)
    original=simulator.read; test_reads=0
    def silent_after_first_test_sample(path,n,rng):
        nonlocal test_reads
        audio,offset=original(path,n,rng)
        if 'test_' in Path(path).name:
            test_reads+=1
            if test_reads>2: audio=audio*0
        return audio,offset
    monkeypatch.setattr(simulator,'read',silent_after_first_test_sample)
    with pytest.raises(ValueError,match=r'split=test, sample_index=1 .*attempts=20'):
        simulate(args)
    assert test_reads==42
    assert not Path(args.output).exists() and not list(tmp_path.glob('.*.incomplete-*'))
