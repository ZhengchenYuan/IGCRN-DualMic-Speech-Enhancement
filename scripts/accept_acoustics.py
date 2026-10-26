"""Numerical propagation comparison, without training or public audio."""
import argparse
import json
from pathlib import Path
import numpy as np
from igcrn.acoustics import make_rirs
from igcrn.v2_common import json_write,new_output
from scripts.diagnose_spatial import theoretical_delay


def inspect_backend(backend,angle):
    a=np.deg2rad(angle)
    scene=dict(room=[5.,5.,3.],mics=[[2.49,2.5,1.5],[2.51,2.5,1.5]],
        sources=[[2.5+1.5*np.sin(a),2.5+1.5*np.cos(a),1.5],[2.5,4.,1.5]],rt60=.25)
    h,meta=make_rirs(scene,backend); fs=16000
    # Direct backend is used for strict free-field tests; truncate legacy only
    # where all reflected paths start later than the isolated direct arrival.
    direct=h[:,0,:100] if backend=='legacy' else h[:,0]
    H=np.fft.rfft(direct,n=16000); tau=theoretical_delay(angle)
    bands={}
    for f in [1000,4000,7000]:
        ipd=np.angle(H[1,f]*np.conj(H[0,f])); expected=-2*np.pi*f*tau/fs
        bands[str(f)]=dict(ipd_error_degrees=float(np.rad2deg(np.angle(np.exp(1j*(ipd-expected))))),
                          ild_db=float(20*np.log10(abs(H[1,f])/abs(H[0,f]))))
    frequencies=np.arange(500,4001)
    phase=np.unwrap(np.angle(H[1,frequencies]*np.conj(H[0,frequencies])))
    inferred=-np.polyfit(2*np.pi*frequencies/fs,phase,1)[0]
    return dict(backend=backend,angle=angle,theory_tdoa_samples=tau,phase_slope_tdoa_samples=float(inferred),
        tdoa_error_samples=float(inferred-tau),bands=bands,metadata=meta,
        caveat='PRA first/full responses include multipath; phase need not follow free-field theory')


def acceptance(output):
    out=new_output(output); out.mkdir(parents=True)
    results=[inspect_backend('legacy',a) for a in [0,22.5,45,90]]
    try:
        import pyroomacoustics
    except ImportError as exc:
        result=dict(status='UNVERIFIED_PRA_UNAVAILABLE',reason=str(exc),cases=results)
        json_write(out/'acoustics.json',result); return result
    for backend in ['pra_direct','pra_first']:
        results.extend(inspect_backend(backend,a) for a in [0,22.5,45,90])
    direct=[r for r in results if r['backend']=='pra_direct']
    errors=[abs(r['tdoa_error_samples']) for r in direct]
    passed=max(errors)<.05 and all(abs(r['bands']['4000']['ipd_error_degrees'])<3 for r in direct)
    # RT60 targets are NOT exact measurements; collect actual modeled decay.
    rt=[]
    for target in [.15,.25,.35]:
        scene=dict(room=[5.,5.,3.],mics=[[2.49,2.5,1.5],[2.51,2.5,1.5]],
            sources=[[3.25,3.799038105676658,1.5],[1.75,3.799038105676658,1.5]],rt60=target)
        h,meta=make_rirs(scene,'pra'); rt.append(meta)
    decay_ok=all(m['measured_rt60_seconds'] is not None and np.isfinite(m['measured_rt60_seconds']).all()
                 and np.min(m['measured_rt60_seconds'])>0 and m['max_order']>1 for m in rt)
    result=dict(status='PASS' if passed and decay_ok else 'FAIL',direct_tdoa_tolerance_samples=.05,
        direct_4khz_ipd_tolerance_degrees=3,cases=results,full_rt60_probes=rt)
    json_write(out/'acoustics.json',result); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    print(acceptance(p.parse_args().output)['status'])
