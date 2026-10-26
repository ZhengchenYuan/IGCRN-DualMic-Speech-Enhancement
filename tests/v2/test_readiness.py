"""Preflight is read-only; proof records below are explicitly control fixtures."""
import json
from pathlib import Path
import pytest
import torch
import yaml
from igcrn.v2_common import json_write,sha
from scripts import check_v2_ready as preflight


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(preflight,'pra_version',lambda:'0.10.0')
    cfg=yaml.safe_load(Path('configs/v2.yaml').read_text())
    manifests=[]
    for split in ('train','valid','test'):
        p=tmp_path/f'{split}.jsonl';p.write_text(json.dumps(dict(split=split,speech={'speaker_id':split},noise={'recording_id':split}))+'\n')
        manifests.append(dict(split=split,path=str(p),sha256=sha(p)))
    cp=tmp_path/'dual.pt';torch.save(dict(mode='dual',epoch=29,manifest_hashes=[r['sha256'] for r in manifests[:2]]),cp)
    catalog=[];roles={'speech':{},'noise':{}}
    for kind in roles:
        for split in ('train','valid','test'):
            source=tmp_path/('source' if kind=='speech' and split=='train' else f'source_{kind}_{split}')
            source.write_text('original fixture '+kind+' '+split)
            roles[kind][split]=split
            catalog.append(dict(kind=kind,speaker_id=split,recording_id=split,split=split,path=str(source),sha256=sha(source)))
    reg=tmp_path/'registry.json'
    json_write(reg,dict(v1_manifests=manifests,v1_checkpoint=dict(sha256=sha(cp),epoch=29),roles=roles,catalog=catalog))
    physical=tmp_path/'physical.json';json_write(physical,dict(status='PASS',full_rt60_probes=[
        dict(scene={'rt60':t},pra_version='0.10.0') for t in (.15,.25,.35)]))
    diag=tmp_path/'diag.json';json_write(diag,dict(status='VERIFIED',requested_cases=1,verified_cases=1,
        checkpoint_sha256=sha(cp),evaluation_manifest_sha256=manifests[1]['sha256']))
    return cfg,reg,physical,diag,cp


def test_ready_preflight_writes_nothing(tmp_path,evidence):
    before={str(p):sha(p) for p in tmp_path.iterdir() if p.is_file()}
    result=preflight.check(*evidence,data=tmp_path/'not_generated',run=tmp_path/'not_started')
    assert result['status']=='READY_FOR_APPROVED_EXECUTION' and not result['training_started']
    assert before=={str(p):sha(p) for p in tmp_path.iterdir() if p.is_file()}
    assert not (tmp_path/'not_generated').exists() and not (tmp_path/'not_started').exists()


@pytest.mark.parametrize('problem',['missing_pra','missing_manifest','mutated_source','different_checkpoint','missing_audio','expired','changed_v1_role'])
def test_preflight_blocks_without_changing_inputs(tmp_path,evidence,monkeypatch,problem):
    cfg,reg,physical,diag,cp=evidence
    run=tmp_path/'run'
    if problem=='missing_pra':monkeypatch.setattr(preflight,'pra_version',lambda:None)
    elif problem=='missing_manifest':(tmp_path/'train.jsonl').unlink()
    elif problem=='mutated_source':(tmp_path/'source').write_bytes(b'corrupt')
    elif problem=='different_checkpoint':torch.save(dict(mode='dual',epoch=28,manifest_hashes=['wrong','wrong']),cp)
    elif problem=='missing_audio':
        report=json.loads(diag.read_text());report.update(status='UNVERIFIED',verified_cases=0);json_write(diag,report)
    elif problem=='changed_v1_role':
        record=json.loads(reg.read_text());record['roles']['speech']['test']='train';json_write(reg,record)
    else:json_write(run/'budget.json',dict(deadline_unix=0,budget_minutes=300))
    before={str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=preflight.check(cfg,reg,physical,diag,cp,run=run)
    assert result['status']=='BLOCKED' and not result['training_started']
    if problem=='changed_v1_role': assert any(r['check']=='protected_v1_roles' and r['status']=='FAIL' for r in result['checks'])
    assert before=={str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
