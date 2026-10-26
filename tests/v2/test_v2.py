"""Small generated fixtures; never public-corpus experimental results."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from igcrn.v2_common import json_write,sha,start_budget,rng_for
from igcrn.acoustics import make_rirs,scene_for,cached_rirs
from igcrn.paired import run_pairs,EpochInterrupted
from scripts.register_v2_split import register
from scripts import simulate_v2
from scripts.diagnose_spatial import signal_stats,theoretical_delay
from scripts.evaluate_v2 import evaluate


@pytest.fixture
def config(): return yaml.safe_load(Path('configs/v2.yaml').read_text())


@pytest.fixture
def registered(tmp_path):
    sources=tmp_path/'sources'; sources.mkdir(); catalog=[]; paths=[]
    for i,split in enumerate(('train','valid','test')):
        for j,kind in enumerate(('speech','noise')):
            path=sources/f'{split}_{kind}.wav'
            sf.write(path,np.random.default_rng(i*2+j).normal(0,.1,80000),16000,subtype='PCM_16')
            catalog.append(dict(path=str(path),kind=kind,split=split,speaker_id=str(i),recording_id=str(i),
                license='original generated test fixture',source_url='fixture',sha256=sha(path)))
        v1=tmp_path/'v1'; v1.mkdir(exist_ok=True); path=v1/f'{split}.jsonl'
        path.write_text(json.dumps(dict(id=split,split=split,speech=catalog[-2],noise=catalog[-1]))+'\n'); paths.append(path)
    cat=tmp_path/'catalog.jsonl'; cat.write_text(''.join(json.dumps(r)+'\n' for r in catalog))
    registry=register(paths,cat,2026); output=tmp_path/'registry.json'; json_write(output,registry)
    return output,cat,paths


def fake_cache(scene,cache,backend):
    # Explicit legacy fixture for fast control tests when PRA is not installed.
    from igcrn.acoustics import scene_key
    h,meta=make_rirs(scene,'legacy'); meta.update(cache_key=scene_key(scene,'legacy-fixture'),sha256='fixture-rir')
    return h,meta


@pytest.fixture
def dataset(tmp_path,config,registered,monkeypatch):
    monkeypatch.setattr(simulate_v2,'cached_rirs',fake_cache)
    registry,_,_=registered; out=tmp_path/'data'
    simulate_v2.simulate(config,registry,out,tmp_path/'cache',counts={'train':3,'valid':3,'test':3})
    return out


@pytest.mark.parametrize('kind,key',[('speech','speaker_id'),('noise','recording_id')])
def test_registry_preserves_v1_and_rejects_leakage(registered,kind,key):
    registry,catalog,paths=registered; original=json.loads(registry.read_text())
    entries=[json.loads(x) for x in catalog.read_text().splitlines()]
    for r in entries: r['split']='train'
    catalog.write_text(''.join(json.dumps(x)+'\n' for x in entries))
    assert register(paths,catalog,2026)['roles']==original['roles']
    with pytest.raises(ValueError,match='three'): register(paths[:1],catalog,2026)
    bad=json.loads(paths[1].read_text()); bad[kind][key]='0'
    paths[1].write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='identity leakage'): register(paths,catalog,2026)


def test_independent_rng_and_scene(config):
    assert np.array_equal(rng_for(1,'test',3).normal(size=10),rng_for(1,'test',3).normal(size=10))
    first=scene_for(config,'test',2)
    for i in range(100): rng_for(1,'train',i).normal(size=100)
    assert scene_for(config,'test',2)==first
    assert first['rt60']==.35 and first['mics']==config['mics']


def test_target_snr_and_determinism(tmp_path,dataset,config,registered,monkeypatch):
    monkeypatch.setattr(simulate_v2,'cached_rirs',fake_cache)
    other=tmp_path/'other'; simulate_v2.simulate(config,registered[0],other,tmp_path/'cache',counts={'train':3,'valid':3,'test':3})
    from scipy.signal import fftconvolve
    for split in ('train','valid','test'):
        a=[json.loads(x) for x in (dataset/f'{split}.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (other/f'{split}.jsonl').read_text().splitlines()]
        for i,(r,s) in enumerate(zip(a,b)):
            assert r['mixture_sha256']==s['mixture_sha256'] and r['target_sha256']==s['target_sha256']
            assert r['speech']['split']==r['noise']['split']==split and r['snr_db']==[-3,0,3][i]
            x,_=sf.read(r['mixture']); y,_=sf.read(r['target'])
            assert abs(10*np.log10(np.mean(y*y)/np.mean((x[:,0]-y)**2))-r['snr_db'])<1e-5
            h,_=fake_cache(r['geometry'],None,None); source,_=sf.read(r['speech']['path'])
            offset=r['speech_offset_16k']; n=r['pre_roll_samples']+32000
            expected=fftconvolve(source[offset:offset+n],h[0,0])[r['crop_start']:r['crop_start']+32000]*r['gain']
            assert np.allclose(y,expected,atol=1e-7)
    with pytest.raises(FileExistsError): simulate_v2.simulate(config,registered[0],dataset,tmp_path/'c',counts={'train':3,'valid':3,'test':3})


def tiny_config(config):
    cfg=copy.deepcopy(config); cfg.update(model=dict(width=8,depth=2,n_fft=512,hop=256),max_epochs=2,
        budget_minutes=10,evaluation_reserve_minutes=0,sync_reserve_minutes=0,safety_minutes=0,initial_pair_estimate_minutes=.01)
    return cfg


def test_paired_resume_matches_continuous(tmp_path,dataset,config):
    cfg=tiny_config(config); a=tmp_path/'continuous'; b=tmp_path/'resumed'
    assert run_pairs(cfg,dataset,a,'cpu',True)=='PAIRS_FINISHED_MAX_EPOCHS'
    assert run_pairs(cfg,dataset,b,'cpu',True,max_pairs_this_call=1)=='PAUSED_BY_CALLER'
    assert run_pairs(cfg,dataset,b,'cpu',True)=='PAIRS_FINISHED_MAX_EPOCHS'
    for mode in ('dual','duplicate'):
        def load(run):
            rec=json.loads((run/'pair.json').read_text()); return torch.load(rec['checkpoints'][mode]['path'],weights_only=False)
        left,right=load(a),load(b)
        assert left['global_step']==right['global_step']==6
        assert all(torch.equal(left['model'][k],right['model'][k]) for k in left['model'])
    assert evaluate(dataset,b,allow_acceptance=True)
    assert len(json.loads((b/'evaluation/summary.json').read_text()))==12
    assert not (b/'experiment_complete.json').exists()  # Only controller may publish completion.


def test_interrupted_pair_is_not_published(tmp_path,dataset,config,monkeypatch):
    import igcrn.paired as paired
    original=paired.one_epoch; calls=[]
    def interrupt(arm,device,deadline):
        calls.append(1)
        if len(calls)==2: raise EpochInterrupted('fixture')
        return original(arm,device,deadline)
    monkeypatch.setattr(paired,'one_epoch',interrupt)
    run=tmp_path/'interrupted'
    assert run_pairs(tiny_config(config),dataset,run,'cpu',True)=='INTERRUPTED_UNCOMMITTED_PAIR'
    assert not (run/'pair.json').exists()
    assert json.loads((run/'state.json').read_text())['completed_epochs']==0


def test_budget_gate_and_not_reset(tmp_path,dataset,config):
    cfg=tiny_config(config); cfg['initial_pair_estimate_minutes']=100
    run=tmp_path/'budget'
    assert run_pairs(cfg,dataset,run,'cpu',True)=='STOPPED_NO_PAIR'
    before=(run/'budget.json').read_bytes()
    start_budget(run/'budget.json',cfg['budget_minutes'])
    assert (run/'budget.json').read_bytes()==before and not (run/'pair.json').exists()
    with pytest.raises(ValueError): start_budget(run/'budget.json',300)


def test_perturbation_statistics():
    rng=np.random.default_rng(22); mono=rng.normal(size=4096); same=np.stack([mono,mono],1)
    stats=signal_stats(same); assert abs(stats['correlation']-1)<1e-10 and stats['difference_energy_ratio']==0
    assert stats['gcc_phat_tdoa_samples']==0 and abs(theoretical_delay(90))<1
    assert abs(theoretical_delay(-45)+theoretical_delay(45))<1e-10


def test_production_guards(tmp_path,dataset,config):
    with pytest.raises(ValueError,match='T4'): run_pairs(config,dataset,tmp_path/'guard','cpu')


def test_pra_physical_acceptance(tmp_path):
    pytest.importorskip('pyroomacoustics')
    from scripts.accept_acoustics import acceptance
    result=acceptance(tmp_path/'physical')
    assert result['status']=='PASS'
    assert len(result['full_rt60_probes'])==3


def test_pra_cache_and_small_real_generation(tmp_path,config,registered):
    pytest.importorskip('pyroomacoustics')
    scene=scene_for(config,'train',0)
    a,ma=cached_rirs(scene,tmp_path/'cache'); b,mb=cached_rirs(scene,tmp_path/'cache')
    fresh,_=make_rirs(scene,'pra')
    assert np.array_equal(a,b) and np.array_equal(a,fresh) and ma==mb
    assert ma['max_order']>1 and ma['global_delay_samples']>0
    out=tmp_path/'real'; other=tmp_path/'real_again'
    counts={'train':3,'valid':3,'test':3}
    simulate_v2.simulate(config,registered[0],out,tmp_path/'cache',counts=counts)
    simulate_v2.simulate(config,registered[0],other,tmp_path/'second_cache',counts=counts)
    from scipy.signal import fftconvolve
    assert json.loads((out/'dataset.json').read_text())['status']=='acceptance_only'
    for split in counts:
        left=[json.loads(x) for x in (out/f'{split}.jsonl').read_text().splitlines()]
        right=[json.loads(x) for x in (other/f'{split}.jsonl').read_text().splitlines()]
        for r,s in zip(left,right):
            assert r['mixture_sha256']==s['mixture_sha256'] and r['target_sha256']==s['target_sha256']
            assert r['rir_sha256']==s['rir_sha256']
            x,_=sf.read(r['mixture']); y,_=sf.read(r['target'])
            assert abs(10*np.log10(np.mean(y*y)/np.mean((x[:,0]-y)**2))-r['snr_db'])<1e-5
            with np.load(tmp_path/'cache'/(r['rir_key']+'.npz')) as f: h=f['rir']
            source,_=sf.read(r['speech']['path']); offset=r['speech_offset_16k']; pre=r['crop_start']
            expected=fftconvolve(source[offset:offset+pre+32000],h[0,0])[pre:pre+32000]*r['gain']
            assert np.max(np.abs(y-expected))<1e-7


def test_frozen_diagnostic_conditions_do_not_modify_inputs(tmp_path,dataset,config):
    from scripts.diagnose_spatial import diagnose
    from igcrn.model import IGCRN
    cfg=tiny_config(config); model=IGCRN(**cfg['model'])
    (tmp_path/'v1model').mkdir()
    checkpoint=tmp_path/'v1model'/'diagnostic.pt'
    torch.save(dict(model=model.state_dict(),mode='dual',config=cfg,epoch=0,manifest_hashes=['fixture','fixture']),checkpoint)
    before=sha(checkpoint)
    result=diagnose(dataset/'valid.jsonl',checkpoint,tmp_path/'diagnosis',limit=1)
    assert result['status']=='VERIFIED' and sha(checkpoint)==before
    assert set(result['cases'][0]['conditions'])=={'normal','duplicated_mic1','zero_mic2'}
    assert result['cases'][0]['conditions']['normal']['output_difference_ratio']==0


def test_acceptance_not_misreported_as_production(dataset):
    from igcrn.paired import verify_dataset
    with pytest.raises(ValueError,match='Acceptance'): verify_dataset(dataset)


def test_atomic_publication_on_simulation_failure(tmp_path,registered,config,monkeypatch):
    def broken(*args,**kwargs): raise RuntimeError('fixture backend failure')
    monkeypatch.setattr(simulate_v2,'cached_rirs',broken)
    out=tmp_path/'failure'
    with pytest.raises(RuntimeError): simulate_v2.simulate(config,registered[0],out,tmp_path/'cache',counts={'train':1,'valid':1,'test':1})
    assert not out.exists() and not list(tmp_path.glob('.*.incomplete-*'))


def test_late_pair_interrupt_preserves_last_complete_pair(tmp_path,dataset,config,monkeypatch):
    import igcrn.paired as paired
    cfg=tiny_config(config); run=tmp_path/'paired'
    run_pairs(cfg,dataset,run,'cpu',True,max_pairs_this_call=1)
    committed=(run/'pair.json').read_bytes(); original=paired.one_epoch; calls=[]
    def stop(arm,device,deadline):
        calls.append(1)
        if len(calls)==2: raise EpochInterrupted('fixture')
        return original(arm,device,deadline)
    monkeypatch.setattr(paired,'one_epoch',stop)
    assert run_pairs(cfg,dataset,run,'cpu',True)=='INTERRUPTED_UNCOMMITTED_PAIR'
    assert (run/'pair.json').read_bytes()==committed
    assert not (run/'experiment_complete.json').exists()


def test_evaluation_timeout_cannot_publish_results(tmp_path,dataset,config):
    import time
    cfg=tiny_config(config); run=tmp_path/'evaluation_timeout'
    run_pairs(cfg,dataset,run,'cpu',True,max_pairs_this_call=1)
    assert evaluate(dataset,run,deadline=time.time()-1,allow_acceptance=True) is False
    assert (run/'evaluation/partial.json').exists()
    assert not (run/'evaluation/summary.json').exists() and not (run/'experiment_complete.json').exists()


def test_changed_manifest_refuses_resume(tmp_path,dataset,config):
    cfg=tiny_config(config); run=tmp_path/'hash_guard'
    run_pairs(cfg,dataset,run,'cpu',True,max_pairs_this_call=1)
    with (dataset/'train.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='manifest changed'): run_pairs(cfg,dataset,run,'cpu',True)


def test_registry_checkpoint_binding(tmp_path,registered):
    registry,catalog,paths=registered
    checkpoint=tmp_path/'binding.pt'
    torch.save(dict(mode='dual',epoch=29,manifest_hashes=[sha(paths[0]),sha(paths[1])]),checkpoint)
    result=register(paths,catalog,2026,checkpoint)
    assert result['v1_checkpoint']['sha256']==sha(checkpoint)
    torch.save(dict(mode='dual',epoch=29,manifest_hashes=['wrong','wrong']),checkpoint)
    with pytest.raises(ValueError,match='do not match'): register(paths,catalog,2026,checkpoint)


def test_controller_requires_real_acceptance_before_generation(tmp_path,config,registered):
    from scripts.run_v2 import execute
    run=tmp_path/'v2'/'blocked'; data=tmp_path/'must_not_generate'
    with pytest.raises(ValueError,match='Physical PRA acceptance'):
        execute(config,data,registered[0],tmp_path/'cache',run)
    assert not data.exists() and not (run/'experiment_complete.json').exists()
    assert json.loads((run/'state.json').read_text())['status']=='FAILED'


def test_preflight_initialization_and_v1_path_guard(tmp_path,config):
    from scripts.run_v2 import initialize_run
    run=tmp_path/'v2'/'run'; start_budget(run/'budget.json',300)
    (run/'preflight.log').write_text('fixture')
    initialize_run(config,tmp_path/'data',run)
    before=(run/'owner.json').read_bytes()
    initialize_run(config,tmp_path/'data',run)
    assert (run/'owner.json').read_bytes()==before
    with pytest.raises(ValueError,match='separate'): initialize_run(config,tmp_path/'data',tmp_path/'v1')


def test_geometry_and_source_streams_and_snr_are_not_confounded(config):
    from igcrn.acoustics import geometry_index
    assert not np.array_equal(rng_for(2026,'train',0,stream=0).random(10),rng_for(2026,'train',0,stream=1).random(10))
    for split,n in config['counts'].items():
        joint=np.zeros((3,3),int); usage={}
        for i in range(n):
            index=geometry_index(config,split,i); usage[index]=usage.get(index,0)+1
            joint[index%3,i%3]+=1
        assert np.all(joint>0)
        assert np.array_equal(joint.sum(0),[n//3]*3) and np.array_equal(joint.sum(1),[n//3]*3)
        assert len(usage)==config['geometry_counts'][split]
        assert set(usage.values())=={n//config['geometry_counts'][split]}


def test_v2_unusable_source_retry_is_bounded(tmp_path,registered,config,monkeypatch):
    monkeypatch.setattr(simulate_v2,'cached_rirs',fake_cache)
    calls=[]
    def empty(*args): calls.append(1); return None,None
    monkeypatch.setattr(simulate_v2,'segment',empty)
    out=tmp_path/'silent'
    with pytest.raises(ValueError,match='split=train, sample_index=0, attempts=20'):
        simulate_v2.simulate(config,registered[0],out,tmp_path/'cache',counts={'train':1,'valid':1,'test':1})
    assert len(calls)==40 and not out.exists()
