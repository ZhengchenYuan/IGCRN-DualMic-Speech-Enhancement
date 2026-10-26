"""Explicit RIR backends; original scripts.simulate.rir is left untouched."""
import json
import time
from pathlib import Path
import numpy as np
from .v2_common import json_write, sha


def scene_key(scene, backend):
    import hashlib
    return hashlib.sha256(json.dumps(dict(scene=scene,backend=backend),sort_keys=True).encode()).hexdigest()


def make_rirs(scene, backend='pra'):
    start=time.monotonic(); positions=scene['sources']; mics=np.asarray(scene['mics'],float)
    meta=dict(backend=backend,scene=scene,speed_of_sound=343.)
    if backend=='legacy':
        from scripts.simulate import rir
        h=[[rir(position,mic,room=scene['room']) for position in positions] for mic in mics]
        meta.update(global_delay_samples=0,max_order=1,measured_rt60_seconds=None)
    else:
        import pyroomacoustics as pra
        if backend=='pra_direct': absorption,order=1.,0
        elif backend=='pra_first': absorption,order=1-.65**2,1
        elif backend=='pra': absorption,order=pra.inverse_sabine(scene['rt60'],scene['room'])
        else: raise ValueError(f'Unknown backend {backend}')
        room=pra.ShoeBox(scene['room'],fs=16000,materials=pra.Material(absorption),
                         max_order=int(order),ray_tracing=False,use_rand_ism=False,air_absorption=False)
        room.set_sound_speed(343.)
        room.add_microphone_array(mics.T)
        for position in positions: room.add_source(position)
        room.compute_rir(); h=room.rir
        measured=None
        if backend=='pra':
            try: measured=np.asarray(room.measure_rt60()).tolist()
            except (ValueError,RuntimeError): pass
        meta.update(pra_version=pra.__version__,energy_absorption=float(absorption),max_order=int(order),
                    global_delay_samples=int(pra.constants.get('frac_delay_length')//2),
                    measured_rt60_seconds=measured)
    lengths=[[len(v) for v in row] for row in h]; length=max(map(max,lengths))
    array=np.zeros((2,2,length),dtype=np.float64)
    for m in range(2):
        for s in range(2): array[m,s,:len(h[m][s])]=h[m][s]
    if not np.isfinite(array).all() or np.any(np.sum(array**2,axis=2)<=0):
        raise ValueError('Non-finite or empty RIR')
    meta.update(original_lengths=lengths,seconds=time.monotonic()-start)
    return array,meta


def cached_rirs(scene, cache, backend='pra'):
    """Both the scene and installed PRA version participate in the cache key."""
    if backend.startswith('pra'):
        import pyroomacoustics as pra
        version=pra.__version__
    else: version='legacy-11b0bb3'
    key=scene_key(dict(scene,implementation_version=version),backend)
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    path=cache/(key+'.npz'); record=cache/(key+'.json')
    if path.exists() or record.exists():
        if not (path.exists() and record.exists()): raise ValueError(f'Incomplete RIR cache: {key}')
        meta=json.loads(record.read_text())
        if sha(path)!=meta['sha256']: raise ValueError(f'Corrupt RIR cache: {key}')
        with np.load(path,allow_pickle=False) as data: h=data['rir'].copy()
        return h,meta
    h,meta=make_rirs(scene,backend)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream: np.savez(stream,rir=h)
    tmp.replace(path); meta.update(cache_key=key,sha256=sha(path))
    json_write(record,meta)
    return h,meta


def scene_for(cfg, split, index):
    from .v2_common import rng_for
    rng=rng_for(cfg['seed'],split,index,stream=1)
    angles=rng.uniform(*cfg['angles_degrees'],size=2)
    sources=[]
    for angle in angles:
        rad=np.deg2rad(angle)
        sources.append([2.5+cfg['source_radius']*np.sin(rad),
                        2.5+cfg['source_radius']*np.cos(rad),cfg['source_height']])
    return dict(room=cfg['room'],mics=cfg['mics'],sources=sources,angles=angles.tolist(),
                rt60=cfg['rt60_seconds'][index%len(cfg['rt60_seconds'])],split=split,geometry_index=index)


def geometry_index(cfg,split,sample_index):
    """Independent shuffled cycles prevent RT60 from being tied to SNR=i%3."""
    from .v2_common import rng_for
    n=cfg['geometry_counts'][split]
    permutation=rng_for(cfg['seed'],split,sample_index//n,stream=2).permutation(n)
    return int(permutation[sample_index%n])
