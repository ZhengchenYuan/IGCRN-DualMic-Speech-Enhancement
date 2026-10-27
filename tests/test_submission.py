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


def test_audio_page_metrics_links_and_original_export_mapping():
    import re
    page = (ROOT / 'examples/README.md').read_text()
    assert all(word not in page.lower() for word in ('blocked', 'unverified', 'missing audio', 'recovery pending'))
    audio_ids = []
    with (ROOT / 'results/v1/per_utterance.csv').open(newline='') as stream:
        metrics = {(r['id'], r['method']): r for r in csv.DictReader(stream)}
    for line in page.splitlines():
        if not line.startswith('| test_'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        case = cells[0]
        audio_ids.append(case)
        assert cells[1] == ('Positive' if float(metrics[case, 'dual']['si_sdri']) > 0 else 'Negative')
        assert int(cells[2]) == int(metrics[case, 'dual']['snr_db'])
        assert cells[-2] == f"{float(metrics[case, 'dual']['si_sdri']):+.4f}"
        assert cells[-1] == f"{float(metrics[case, 'duplicate']['si_sdri']):+.4f}"
    assert audio_ids == ['test_00000', 'test_00001', 'test_00002', 'test_00003']
    for link in re.findall(r'\]\(([^)]+)\)', page):
        if not link.startswith('http'):
            assert (ROOT / 'examples' / link).exists(), link
    # Verify byte identity against the independently retained original V1 inventory.
    old_inventory = json.loads((ROOT / 'docs/v2/v1_assets_integrity.json').read_text())
    original = [json.loads(s) for s in (ROOT / 'manifests/evaluation_manifest.jsonl').read_text().splitlines()]
    exported = json.loads((ROOT / 'examples/v1/manifest.json').read_text())['cases']
    assert [r['id'] for r in exported] == [r['id'] for r in original[:4]]
    assert all(Path(e['path']).parent.name == r['id'] for r in exported for e in r['files'].values())
    for case in exported:
        for method, entry in case['files'].items():
            original_name = f"evaluation/{case['id']}_{method}.wav"
            assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == old_inventory['files'][original_name]


def test_existing_waveforms_reproduce_their_own_csv_rows():
    from scripts.evaluate import sisdr
    from pystoi import stoi
    with (ROOT / 'results/v1/per_utterance.csv').open(newline='') as stream:
        expected = {(r['id'], r['method']): r for r in csv.DictReader(stream)}
    cases = json.loads((ROOT / 'examples/v1/manifest.json').read_text())['cases']
    for case in cases:
        waveforms = {method: sf.read(ROOT / entry['path'], dtype='float32')[0]
                     for method, entry in case['files'].items()}
        target = waveforms['target']
        baseline = sisdr(waveforms['mic1'], target)
        for method in ('mic1', 'dual', 'duplicate'):
            actual = {'si_sdr': sisdr(waveforms[method], target),
                      'si_sdri': sisdr(waveforms[method], target) - baseline,
                      'stoi': float(stoi(target, waveforms[method], 16000, extended=False))}
            for metric, value in actual.items():
                assert np.isclose(value, float(expected[case['id'], method][metric]),
                                  atol=1e-6, rtol=0), (case['id'], method, metric)


def test_quantitative_case_analysis_matches_csv_and_audio_evidence():
    import re
    page = (ROOT / 'results/v1/CASE_ANALYSIS.md').read_text()
    with (ROOT / 'results/v1/per_utterance.csv').open(newline='') as stream:
        metrics = {(r['id'], r['method']): r for r in csv.DictReader(stream)}
    ids = []
    for line in page.splitlines():
        if not line.startswith('| test_'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        case = cells[0]
        ids.append(case)
        dual = float(metrics[case, 'dual']['si_sdri'])
        duplicate = float(metrics[case, 'duplicate']['si_sdri'])
        assert cells[1] == ('Positive' if dual > 0 else 'Negative')
        assert int(cells[2]) == int(metrics[case, 'dual']['snr_db'])
        assert cells[3:6] == [f'{dual:+.4f}', f'{duplicate:+.4f}', f'{dual-duplicate:+.4f}']
        if case == 'test_00003':
            assert '[Audio and metrics]' in cells[6]
        else:
            assert cells[6] == 'Quantitative only'
            assert not (ROOT / 'examples' / case).exists()
    assert ids == ['test_00067', 'test_00105', 'test_00003', 'test_00096']
    assert '120-case' in page
    for path in (ROOT / 'README.md', ROOT / 'examples/README.md', ROOT / 'results/v1/CASE_ANALYSIS.md'):
        content = path.read_text()
        assert all(word not in content.lower() for word in ('blocked', 'unverified', 'missing audio', 'recovery pending'))
        for link in re.findall(r'\]\(([^)]+)\)', content):
            if not link.startswith(('http:', 'https:', '#', 'mailto:')):
                assert (path.parent / link.split('#')[0]).exists(), (path, link)
