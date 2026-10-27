"""Selected-ID export acceptance using explicit synthetic fixtures, no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.io import wavfile
from pystoi import stoi

from scripts import export_examples as exporter


@pytest.mark.parametrize('existing_file', [False, True])
def test_evaluation_preserves_existing_results_before_loading_models(tmp_path, existing_file):
    from scripts.evaluate import main as evaluate
    out = tmp_path / 'evaluation'
    if existing_file:
        out.write_bytes(b'original output file')
        original = out
    else:
        out.mkdir()
        original = out / 'summary.json'
        original.write_bytes(b'original saved evaluation')
    before = original.read_bytes()
    with pytest.raises(FileExistsError, match='new or empty'):
        evaluate(argparse.Namespace(output=out))
    assert original.read_bytes() == before


def test_selection_uses_ids_not_positions():
    items = [{'id': 'test_00003'}, {'id': 'test_00067'}]
    assert exporter.select_indices(items, ['test_00067', 'test_00003']) == [1, 0]
    with pytest.raises(ValueError, match='Unknown'):
        exporter.select_indices(items, ['test_00105'])
    with pytest.raises(ValueError, match='Duplicate'):
        exporter.select_indices(items, ['test_00003', 'test_00003'])
    with pytest.raises(ValueError, match='Duplicate'):
        exporter.select_indices(items * 2, ['test_00003'])


@pytest.mark.parametrize('bad_metrics', [False, True])
def test_export_publication_hashes_and_metric_guard(tmp_path, monkeypatch, bad_metrics):
    rng = np.random.default_rng(4)
    target = rng.normal(size=32000).astype(np.float32) * .05
    x = target + rng.normal(size=32000).astype(np.float32) * .02
    mix = tmp_path / 'mix.wav'; clean = tmp_path / 'target.wav'
    wavfile.write(mix, 16000, np.stack([x, x], axis=1))
    wavfile.write(clean, 16000, target)
    row = {'id': 'test_00067', 'snr_db': 0, 'mixture': str(mix), 'target': str(clean),
           'mixture_sha256': exporter.digest(mix), 'target_sha256': exporter.digest(clean),
           'speech': {'source_id': 'fixture-speech'}, 'noise': {'source_id': 'fixture-noise'}}
    manifest = tmp_path / 'test.jsonl'; manifest.write_text(json.dumps(row) + '\n')
    score = exporter.sisdr(x, target)
    metric_path = tmp_path / 'metrics.csv'
    with metric_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id','snr_db','method','si_sdr','si_sdri','stoi'])
        writer.writeheader()
        for mode in ('mic1', 'dual', 'duplicate'):
            writer.writerow(dict(id=row['id'], snr_db=0, method=mode, si_sdr=score,
                                 si_sdri=1 if bad_metrics else 0, stoi=stoi(target, x, 16000)))
    class Identity(torch.nn.Module):
        def forward(self, audio):
            return audio[:, 0]
    monkeypatch.setattr(exporter, 'IGCRN', lambda **kwargs: Identity())
    provenance = {'evaluation_manifest_sha256': exporter.digest(manifest)}
    paths = {}
    for mode in ('dual', 'duplicate'):
        paths[mode] = tmp_path / f'{mode}.pt'
        torch.save({'mode': mode, 'epoch':29, 'batch_size':1, 'manifest_hashes':['fixture'],
                    'config': {'model':{}}, 'model': {}}, paths[mode])
        provenance[mode] = {'checkpoint_sha256':exporter.digest(paths[mode]), 'epoch':29}
    prov = tmp_path / 'provenance.json'; prov.write_text(json.dumps(provenance))
    out = tmp_path / 'exports'
    args = argparse.Namespace(manifest=manifest, metrics=metric_path, provenance=prov,
                              output=out, ids=[row['id']], **paths)
    if bad_metrics:
        with pytest.raises(ValueError, match='differs from official CSV'):
            exporter.main(args)
        assert not out.exists()
        assert not list(tmp_path.glob('.exports.incomplete-*'))
    else:
        exporter.main(args)
        record = json.loads((out/'export_provenance.json').read_text())['cases'][0]
        assert record['id'] == row['id']
        for entry in record['files'].values():
            file = out / entry['path']; audio, rate = sf.read(file)
            assert audio.shape == (32000,) and rate == 16000
            assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256']
        with pytest.raises(FileExistsError):
            exporter.main(args)
    assert exporter.digest(manifest) == provenance['evaluation_manifest_sha256']
