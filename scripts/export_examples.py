"""Export only explicit V1 test IDs, without changing official metric files."""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi
from scipy.io import wavfile

from igcrn.data import Mixtures
from igcrn.model import IGCRN
from scripts.evaluate import sisdr


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_indices(items, ids):
    keys = [r['id'] for r in items]
    if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate test IDs')
    missing = set(ids) - set(keys)
    if missing:
        raise ValueError(f'Unknown test IDs: {sorted(missing)}')
    return [keys.index(key) for key in ids]


def main(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing existing output: {out}')
    ds = Mixtures(a.manifest)
    indices = select_indices(ds.items, a.ids)
    provenance = json.loads(Path(a.provenance).read_text())
    if digest(a.manifest) != provenance['evaluation_manifest_sha256']:
        raise ValueError('Original evaluation manifest hash mismatch')
    with Path(a.metrics).open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    expected = {(r['id'], r['method']): r for r in metrics}
    if len(expected) != len(metrics):
        raise ValueError('Duplicate metric rows')
    models = {}
    reference = None
    torch.set_num_threads(2)
    for mode in ('dual', 'duplicate'):
        path = getattr(a, mode)
        if digest(path) != provenance[mode]['checkpoint_sha256']:
            raise ValueError(f'{mode} checkpoint hash mismatch')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck['mode'] != mode or ck['epoch'] != provenance[mode]['epoch']:
            raise ValueError(f'{mode} checkpoint identity mismatch')
        if reference is not None:
            for key in ('config', 'epoch', 'batch_size', 'manifest_hashes'):
                if ck[key] != reference[key]:
                    raise ValueError(f'Unequal checkpoint setting: {key}')
        reference = ck
        models[mode] = IGCRN(**ck['config']['model']).eval()
        models[mode].load_state_dict(ck['model'])
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.incomplete-', dir=out.parent))
    records = []
    try:
        with torch.inference_mode():
            for index in indices:
                row = ds.items[index]
                x, y = ds[index]  # Verifies original mixture/target file hashes.
                target = y.numpy()
                if x.shape != (2, 32000) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError(f'Invalid synchronized 2-second audio: {row["id"]}')
                estimates = {'mic1': x[0].numpy()}
                for mode, model in models.items():
                    inp = x if mode == 'dual' else x[[0, 0]]
                    estimates[mode] = model(inp[None])[0].numpy()
                baseline = sisdr(estimates['mic1'], target)
                measured = {}
                for method, audio in estimates.items():
                    score = sisdr(audio, target)
                    measured[method] = {'si_sdr': score, 'si_sdri': score - baseline,
                                        'stoi': float(stoi(target, audio, 16000, extended=False))}
                    saved = expected[row['id'], method]
                    if float(saved['snr_db']) != row['snr_db']:
                        raise ValueError('CSV/manifest SNR mismatch')
                    for metric, value in measured[method].items():
                        tolerance = 1e-4 if metric == 'stoi' else 1e-3
                        if not np.isfinite(value) or abs(value - float(saved[metric])) > tolerance:
                            raise ValueError(f'{row["id"]}/{method}/{metric} differs from official CSV; '
                                             'check numerical environment; no exports published')
                folder = stage / row['id']
                folder.mkdir()
                files = {}
                for method, audio in {'target': target, **estimates}.items():
                    if audio.shape != (32000,) or not np.isfinite(audio).all():
                        raise ValueError('Invalid model output')
                    path = folder / f'{method}.wav'
                    wavfile.write(path, 16000, audio.astype(np.float32))
                    files[method] = {'path': f'{row["id"]}/{method}.wav', 'sha256': digest(path)}
                records.append({'id': row['id'], 'snr_db': row['snr_db'],
                                'speech': row['speech'], 'noise': row['noise'],
                                'mixture_sha256': row['mixture_sha256'],
                                'target_sha256': row['target_sha256'], 'files': files,
                                'measured_metrics': measured})
        (stage / 'export_provenance.json').write_text(json.dumps({
            'original_evaluation_provenance': provenance,
            'metrics_sha256': digest(a.metrics), 'cases': records,
            'note': 'New inference exports, not replacements for original evaluation files. '
                    'Numerical differences beyond 0.001 dB / 0.0001 STOI abort publication.'}, indent=2) + '\n')
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--metrics', default='results/v1/per_utterance.csv')
    p.add_argument('--provenance', default='results/v1/provenance.json')
    p.add_argument('--dual', required=True)
    p.add_argument('--duplicate', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--output', required=True)
    main(p.parse_args())
