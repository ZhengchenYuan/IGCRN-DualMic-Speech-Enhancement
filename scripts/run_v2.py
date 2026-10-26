"""Opt-in 300-minute V2 controller. Default is plan-only, never training."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import yaml
from igcrn.v2_common import json_write,sha,start_budget
from igcrn.paired import run_pairs,verify_dataset
from scripts.simulate_v2 import simulate,approved
from scripts.evaluate_v2 import evaluate


def initialize_run(cfg,data,run):
    approved(cfg); run=Path(run).resolve(); data=Path(data).resolve()
    if 'v2' not in run.parts: raise ValueError('Production run must live in a separate .../v2/<run_id> directory')
    owner=dict(schema_version=2,config=cfg,data=str(data))
    if run.exists() and any(run.iterdir()) and not {p.name for p in run.iterdir()}<={'budget.json','preflight.log'}:
        if not (run/'owner.json').exists() or json.loads((run/'owner.json').read_text())!=owner:
            raise ValueError('Refuse to reuse a V1 or unrelated run directory')
    else: json_write(run/'owner.json',owner)
    return run


def execute(cfg,data,registry,cache,run):
    run=initialize_run(cfg,data,run); data=Path(data).resolve()
    budget=start_budget(run/'budget.json',cfg['budget_minutes'])
    if (run/'experiment_complete.json').exists(): return 'ALREADY_COMPLETE'
    if time.time()>=budget['deadline_unix']:
        json_write(run/'state.json',dict(status='STOPPED_EXPIRED_BUDGET')); return 'STOPPED_EXPIRED_BUDGET'
    json_write(run/'config.json',cfg)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=dict(commit=commit,registry_sha256=sha(registry))
    if (run/'code.json').exists() and json.loads((run/'code.json').read_text())!=code:
        raise ValueError('Resume code/registry provenance changed')
    json_write(run/'code.json',code)
    try:
        physical=run/'acoustic_acceptance'/'acoustics.json'
        diagnostic=run/'v1_diagnostic'/'diagnostic.json'
        if not physical.exists() or json.loads(physical.read_text())['status']!='PASS':
            raise ValueError('Physical PRA acceptance must pass before production execution')
        if not diagnostic.exists() or json.loads(diagnostic.read_text())['status']!='VERIFIED':
            raise ValueError('Actual V1 dual-channel diagnostic is required before production execution')
        if not json.loads(Path(registry).read_text()).get('v1_checkpoint'):
            raise ValueError('Registry must bind all original splits to the V1 checkpoint')
        if not data.exists():
            json_write(run/'state.json',dict(status='PREPARING'))
            # Leave at least one full conservative pair plus evaluation and finalization.
            reserve=60*(cfg['evaluation_reserve_minutes']+cfg['sync_reserve_minutes']+cfg['safety_minutes']+cfg['initial_pair_estimate_minutes'])
            simulate(cfg,registry,data,cache,deadline=budget['deadline_unix']-reserve)
        info=verify_dataset(data)
        if info['config']!=cfg or info['registry_sha256']!=sha(registry): raise ValueError('Existing V2 data config/registry mismatch')
        for file in ['dataset.json','train.jsonl','valid.jsonl','test.jsonl']:
            import shutil
            shutil.copyfile(data/file,run/file)
            if sha(run/file)!=sha(data/file): raise IOError('Manifest/metadata synchronization verification failed')
        shutil.copyfile(registry,run/'registry.json')
        if sha(registry)!=sha(run/'registry.json'): raise IOError('Registry synchronization verification failed')
        status=run_pairs(cfg,data,run)
        if status not in ('PAIRS_FINISHED_MAX_EPOCHS','PAIRS_FINISHED_TIME_BUDGET'): return status
        json_write(run/'state.json',dict(status='EVALUATING'))
        if not evaluate(data,run,budget['deadline_unix']-cfg['sync_reserve_minutes']*60):
            json_write(run/'state.json',dict(status='EVALUATION_INCOMPLETE')); return 'EVALUATION_INCOMPLETE'
        pair=json.loads((run/'pair.json').read_text())
        artifacts={str(p.relative_to(run)):sha(p) for p in (run/'evaluation').iterdir() if p.is_file()}
        if time.time()>=budget['deadline_unix']:
            json_write(run/'state.json',dict(status='FINALIZATION_INCOMPLETE')); return 'FINALIZATION_INCOMPLETE'
        status='COMPLETE' if pair['completed_epochs']==cfg['max_epochs'] else 'COMPLETE_REDUCED_BUDGET'
        json_write(run/'experiment_complete.json',dict(status=status,requested_epochs=cfg['max_epochs'],
            completed_epochs=pair['completed_epochs'],global_step=pair['global_step'],case_count=300,
            paired_checkpoint=pair['checkpoints'],artifacts=artifacts,wall_seconds=time.time()-budget['started_unix']))
        json_write(run/'state.json',dict(status=status)); return status
    except (KeyboardInterrupt,TimeoutError) as exc:
        json_write(run/'state.json',dict(status='STOPPED',reason=str(exc))); return 'STOPPED'
    except Exception as exc:
        json_write(run/'state.json',dict(status='FAILED',reason=str(exc))); raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/v2.yaml')
    p.add_argument('--data',required=True); p.add_argument('--registry',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--run',required=True); p.add_argument('--execute',action='store_true')
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); approved(cfg)
    if not a.execute:
        print(json.dumps(dict(status='PLAN_ONLY',max_epochs=cfg['max_epochs'],budget_minutes=cfg['budget_minutes'],training_started=False)))
        return
    print(execute(cfg,a.data,a.registry,a.cache,a.run))

if __name__=='__main__': main()
