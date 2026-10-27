"""Read-only consistency checks for the supplied V1 delivery artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def test_original_v1_provenance_and_artifact_hashes():
    provenance = json.loads((ROOT / 'results/v1/provenance.json').read_text())
    manifest = ROOT / 'manifests/evaluation_manifest.jsonl'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == provenance['evaluation_manifest_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 120
    assert len({r['id'] for r in rows}) == 120
    assert all(r['split'] == 'test' for r in rows)
    for snr in (-3, 0, 3):
        assert sum(r['snr_db'] == snr for r in rows) == 40
    inventory = json.loads((ROOT / 'results/v1/artifact_sha256.json').read_text())
    for relative, expected in inventory.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


def test_v1_summary_matches_per_utterance_and_same_cases():
    with (ROOT / 'results/v1/per_utterance.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((ROOT / 'results/v1/summary.json').read_text())
    assert len(rows) == 360
    cases = {json.loads(line)['id'] for line in (ROOT / 'manifests/evaluation_manifest.jsonl').read_text().splitlines()}
    for method in ('mic1', 'dual', 'duplicate'):
        selected = [r for r in rows if r['method'] == method]
        assert len(selected) == 120
        assert {r['id'] for r in selected} == cases
    baseline = {r['id']: float(r['si_sdr']) for r in rows if r['method'] == 'mic1'}
    for row in rows:
        assert np.isclose(float(row['si_sdri']), float(row['si_sdr']) - baseline[row['id']], atol=1e-10)
    for group in summary:
        selected = [r for r in rows if r['method'] == group['method'] and
                    (group['snr_db'] == 'all' or int(r['snr_db']) == group['snr_db'])]
        assert len(selected) == group['n']
        for metric in ('si_sdr', 'si_sdri', 'stoi'):
            assert np.isclose(np.mean([float(r[metric]) for r in selected]), group[metric], atol=1e-10)


def test_real_audio_exports_have_matching_shapes_and_attribution():
    examples = json.loads((ROOT / 'examples/v1/manifest.json').read_text())['cases']
    assert len(examples) == 4
    original = {r['id']: r for r in map(json.loads, (ROOT / 'manifests/evaluation_manifest.jsonl').read_text().splitlines())}
    for case in examples:
        source = original[case['id']]
        assert case['snr_db'] == source['snr_db']
        for kind in ('speech', 'noise'):
            assert case[kind] == source[kind]
            assert case[kind]['license'] and case[kind]['source_url']
        assert set(case['files']) == {'mic1', 'target', 'dual', 'duplicate'}
        for entry in case['files'].values():
            audio, rate = sf.read(ROOT / entry['path'])
            assert rate == 16000 and audio.shape == (32000,)
            assert np.isfinite(audio).all()
            assert np.mean(audio ** 2) > 1e-12
