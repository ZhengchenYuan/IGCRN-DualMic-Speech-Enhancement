"""Small public-data preparation. Run as ``python -m scripts.prepare_data``.

Only Mini LibriSpeech train-clean-5 and individually selected ESC-10 clips are
used. Licenses are verified against publisher evidence before audio downloads.
No credentials, full LibriSpeech corpus, training, or Drive audio copies.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath

SLR_PAGE = 'https://www.openslr.org/31/'
SLR_BASE = 'https://www.openslr.org/resources/31/'
SPEECH_ARCHIVE = 'train-clean-5.tar.gz'
# Published by OpenSLR, not generated from a local or synthetic archive.
SPEECH_MD5 = '5df7d4e78065366204ca6845bb08f490'
ESC_COMMIT = '33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
ESC_BASE = f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_COMMIT}/'
SPLITS = ('train', 'valid', 'test')
CATEGORIES = {'rain', 'sea_waves', 'crackling_fire', 'helicopter', 'chainsaw', 'clock_tick'}
PRESETS = {
    'smoke': dict(speakers=(1, 1, 1), utterances_per_speaker=2, noises=(2, 1, 1)),
    'experiment': dict(speakers=(12, 3, 3), utterances_per_speaker=20, noises=(40, 10, 10)),
}

class PreparationError(RuntimeError):
    """An actionable download, integrity, identity, or licensing failure."""


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def download(url, path, expected=None, algorithm='sha256'):
    """Verified cache, streamed download, retry and atomic replacement.

    SHA256 receipts detect local corruption; they are not publisher checksums.
    Speech additionally requires OpenSLR's published MD5. ESC URLs pin a commit.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + '.download.json')
    if path.exists():
        if expected and digest(path, algorithm) != expected:
            raise PreparationError(f'Checksum mismatch: {path}. Remove it and rerun.')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['url'] != url or digest(path) != saved['sha256']:
                raise PreparationError(f'Cached source changed: {path}. Remove it and rerun.')
            return saved
        if not expected:
            raise PreparationError(f'Unverified cached file: {path}. Remove it and rerun.')
    else:
        part = path.with_name(path.name + '.part')
        for attempt in range(3):
            try:
                print(f'Downloading {url}', flush=True)
                req = urllib.request.Request(url, headers={'User-Agent': 'IGCRN-public-data/1.0'})
                with urllib.request.urlopen(req, timeout=60) as response, part.open('wb') as dst:
                    shutil.copyfileobj(response, dst, length=1024 * 1024)
                if expected and digest(part, algorithm) != expected:
                    raise PreparationError(f'Publisher checksum mismatch for {url}; no extraction performed.')
                part.replace(path)
                break
            except (PreparationError, KeyboardInterrupt):
                part.unlink(missing_ok=True)
                raise
            except (OSError, ValueError) as exc:
                part.unlink(missing_ok=True)
                if attempt == 2:
                    raise PreparationError(f'Cannot download {url}: {exc}. No catalog published.') from exc
                time.sleep(attempt + 1)
    info = dict(url=url, sha256=digest(path), bytes=path.stat().st_size)
    if expected:
        info['publisher_checksum'] = dict(algorithm=algorithm, value=expected)
    write_json(receipt, info)
    return info


def verify_licenses(slr_page, esc_license, md5_text):
    """Fail closed; never infer a license from a dataset name or file extension."""
    slr = re.sub('<[^>]+>', ' ', slr_page)
    if not re.search(r'License:\s*CC BY 4\.0', slr):
        raise PreparationError(f'Cannot verify Mini LibriSpeech CC BY 4.0 at {SLR_PAGE}. '
                               'Manually confirm publisher terms before updating the pipeline.')
    flat = ' '.join(esc_license.split())
    if ('The ESC-10 subset is licensed as a Creative Commons Attribution 3.0 Unported' not in flat
            or 'https://creativecommons.org/licenses/by/3.0/' not in flat):
        raise PreparationError('Cannot verify ESC-10 CC BY 3.0 in pinned upstream LICENSE. '
                               'Manual publisher confirmation is required; do not assume ESC-50 has this license.')
    checksums = {p[1].lstrip('*'): p[0] for line in md5_text.splitlines() if len(p := line.split()) == 2}
    if checksums.get(SPEECH_ARCHIVE) != SPEECH_MD5:
        raise PreparationError('OpenSLR checksum evidence changed. Manually verify train-clean-5 release.')


def ordered(values, seed, namespace):
    return sorted(values, key=lambda x: (hashlib.sha256(f'{seed}:{namespace}:{x}'.encode()).hexdigest(), x))


def assign_groups(groups, counts, seed, namespace):
    ids = ordered(groups, seed, namespace)
    if len(ids) < sum(counts):
        raise PreparationError(f'Need {sum(counts)} disjoint {namespace} IDs, found {len(ids)}. Reduce counts.')
    result = {}; start = 0
    for split, count in zip(SPLITS, counts):
        for identity in ids[start:start + count]:
            result[identity] = split
        start += count
    return result


def speech_index(archive):
    """Index regular FLACs without trusting or extracting arbitrary tar paths."""
    groups = defaultdict(list)
    for member in archive.getmembers():
        p = PurePosixPath(member.name)
        if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
            raise PreparationError(f'Unsafe archive member: {member.name}')
        if not member.isfile() or not member.name.endswith('.flac'):
            continue
        match = re.fullmatch(r'LibriSpeech/train-clean-5/(\d+)/(\d+)/(\d+)-(\d+)-(\d+)\.flac', member.name)
        if not match or match[1] != match[3] or match[2] != match[4]:
            raise PreparationError(f'Unexpected speech identity/path: {member.name}')
        groups[match[1]].append(member.name)
    if not groups:
        raise PreparationError('No train-clean-5 FLACs found in the verified speech archive.')
    return groups


def noise_candidates(metadata, license_text):
    """Keep ESC-10 environmental classes; preserve literal upstream attribution."""
    entries = {}
    for line in license_text.splitlines():
        match = re.match(r'- \[([^]]+)\]:\s+clip derived from (.+) \((https?://[^)]+)\) by (.+) \[([^]]+)\]\s*$', line)
        if match:
            entries[match[1]] = dict(original_title=match[2], original_source_url=match[3],
                                    attribution_author=match[4], original_license_label=match[5],
                                    attribution_text=line)
    candidates = defaultdict(list)
    seen = set()
    for r in csv.DictReader(io.StringIO(metadata)):
        if r['esc10'] != 'True' or r['category'] not in CATEGORIES:
            continue
        filename = r['filename']
        if not re.fullmatch(r'\d+-\d+-[A-Z]-\d+\.wav', filename) or filename in seen:
            raise PreparationError(f'Invalid or duplicate ESC filename: {filename}')
        seen.add(filename)
        if filename.split('-')[1] != r['src_file']:
            raise PreparationError(f'Conflicting original noise ID: {filename}')
        # Upstream attribution names refer to the historical .ogg clip, without class suffix.
        key = '-'.join(filename.split('-')[:3]) + '.ogg'
        attribution = entries.get(key)
        if not attribution or attribution['original_license_label'] not in {'CC0', 'CC-BY'}:
            raise PreparationError(f'Cannot verify permitted upstream attribution for {filename}. '
                                   'Manually confirm its original license; no catalog will be published.')
        candidates[r['src_file']].append(dict(r, **attribution))
    return candidates


def audio_info(path):
    import soundfile as sf
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels != 1:
        raise PreparationError(f'Expected nonempty mono source: {path}')
    return dict(sample_rate=info.samplerate, frames=info.frames, channels=info.channels)


def build_catalog(archive_path, metadata, license_text, output, seed, speakers, utterances, noises):
    """Extract only selected speech, download selected noise, retain original bytes."""
    output = Path(output)
    result = []
    noise = noise_candidates(metadata, license_text)  # Check all eligible licenses before extracting audio.
    noise_splits = assign_groups(noise, noises, seed, 'noise-recordings')
    # Index once, then extract selected files in archive order. Random gzip seeks
    # per utterance would repeatedly decompress the archive and be slow in Colab.
    with tarfile.open(archive_path, 'r:gz') as archive:
        speech = speech_index(archive)
    speech_splits = assign_groups(speech, speakers, seed, 'speakers')
    selected = {}
    for speaker, split in speech_splits.items():
        for name in ordered(speech[speaker], seed, f'utterances:{speaker}')[:utterances]:
            selected[name] = (speaker, split)
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if member.name not in selected:
                continue
            member_name = member.name
            speaker, split = selected[member_name]
            target = output/'raw'/'speech'/member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.with_suffix('.part').open('wb') as dst:
                shutil.copyfileobj(src, dst)
            target.with_suffix('.part').replace(target)
            result.append(dict(path=str(target.resolve()), kind='speech', split=split,
                speaker_id=speaker, source_id=Path(member_name).stem, archive_member=member_name,
                dataset='Mini LibriSpeech train-clean-5', source_url=SLR_BASE+SPEECH_ARCHIVE,
                original_source_id=Path(member_name).stem, license='CC-BY-4.0',
                license_url='https://creativecommons.org/licenses/by/4.0/', license_evidence_url=SLR_PAGE,
                attribution='Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; LibriSpeech/OpenSLR',
                selection_seed=seed, sha256=digest(target), **audio_info(target)))
    for recording, split in noise_splits.items():
        # Exactly one derived clip per original recording, even when upstream has multiple takes.
        r = sorted(noise[recording], key=lambda x: x['filename'])[0]
        target = output/'raw'/'noise'/r['filename']; url = ESC_BASE+'audio/'+r['filename']
        download(url, target)
        result.append(dict(path=str(target.resolve()), kind='noise', split=split, recording_id=recording,
            source_id=r['filename'], original_source_id=recording, dataset='ESC-10 environmental subset',
            source_url=url, source_revision=ESC_COMMIT, category=r['category'], upstream_fold=int(r['fold']),
            license='CC-BY-3.0', license_url='https://creativecommons.org/licenses/by/3.0/',
            license_evidence_url=ESC_BASE+'LICENSE', selection_seed=seed, sha256=digest(target),
            **{k: r[k] for k in ('original_title', 'original_source_url', 'attribution_author',
                                 'original_license_label', 'attribution_text')}, **audio_info(target)))
    hashes = [r['sha256'] for r in result]
    if len(set(hashes)) != len(hashes):
        raise PreparationError('Duplicate source audio bytes across selections; catalog not published.')
    return sorted(result, key=lambda r: (SPLITS.index(r['split']), r['kind'], r['source_id']))


def prepare(args):
    preset = PRESETS[args.mode]
    speakers = tuple(args.speakers or preset['speakers'])
    noises = tuple(args.noise_recordings or preset['noises'])
    utterances = (args.utterances_per_speaker if args.utterances_per_speaker is not None
                  else preset['utterances_per_speaker'])
    if any(n <= 0 for n in (*speakers, *noises, utterances)) or args.seed < 0:
        raise PreparationError('Counts must be positive and seed must be nonnegative.')
    output = Path(args.output).resolve()
    # Refuse Drive for raw data. Artifacts can be copied there after preparation.
    if 'drive' in [part.lower() for part in output.parts]:
        raise PreparationError('Raw data belongs under /content/, not Google Drive. Use --output /content/igcrn_data.')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(schema_version=1, mode=args.mode, selection_seed=args.seed, speakers=list(speakers),
                  noise_recordings=list(noises), utterances_per_speaker=utterances,
                  speech_archive=SLR_BASE+SPEECH_ARCHIVE, esc_revision=ESC_COMMIT,
                  noise_categories=sorted(CATEGORIES), selection='sha256(seed:namespace:identity) ordering')
    run_path = output/'preparation.json'
    if run_path.exists() and json.loads(run_path.read_text())['config'] != config:
        raise PreparationError('Output already contains a different selection. Use a new --output directory.')
    evidence = output/'evidence'; evidence.mkdir(exist_ok=True)
    sources = {}
    for name, url in [('slr31.html', SLR_PAGE), ('md5sum.txt', SLR_BASE+'md5sum.txt'),
                       ('ESC-LICENSE', ESC_BASE+'LICENSE'), ('esc50.csv', ESC_BASE+'meta/esc50.csv')]:
        sources[name] = download(url, evidence/name)
    verify_licenses((evidence/'slr31.html').read_text(), (evidence/'ESC-LICENSE').read_text(),
                    (evidence/'md5sum.txt').read_text())
    # Resolve every eligible clip's attribution before downloading any audio.
    assign_groups(noise_candidates((evidence/'esc50.csv').read_text(),
                                  (evidence/'ESC-LICENSE').read_text()), noises, args.seed, 'noise-recordings')
    if args.speech_archive:
        archive = Path(args.speech_archive).resolve()
        if digest(archive, 'md5') != SPEECH_MD5:
            raise PreparationError('Provided speech archive does not match published train-clean-5 MD5.')
        sources['speech'] = dict(url=SLR_BASE+SPEECH_ARCHIVE, sha256=digest(archive),
                                 publisher_checksum=dict(algorithm='md5', value=SPEECH_MD5))
    else:
        archive = output/'downloads'/SPEECH_ARCHIVE
        sources['speech'] = download(SLR_BASE+SPEECH_ARCHIVE, archive, SPEECH_MD5, 'md5')
    catalog = build_catalog(archive, (evidence/'esc50.csv').read_text(), (evidence/'ESC-LICENSE').read_text(),
                            output, args.seed, speakers, utterances, noises)
    # Never publish a partly prepared catalog.
    tmp = output/'catalog.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in catalog))
    tmp.replace(output/'catalog.jsonl')
    summary = {split: {kind: sum(r['split']==split and r['kind']==kind for r in catalog)
                       for kind in ('speech', 'noise')} for split in SPLITS}
    write_json(run_path, dict(config=config, sources=sources, counts=summary,
                             catalog_sha256=digest(output/'catalog.jsonl')))
    print(json.dumps(dict(catalog=str(output/'catalog.jsonl'), counts=summary), indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/content/igcrn_data', help='Local raw-data root; never a Drive directory')
    p.add_argument('--mode', choices=PRESETS, default='smoke')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--speakers', type=int, nargs=3, metavar=('TRAIN', 'VALID', 'TEST'))
    p.add_argument('--noise-recordings', type=int, nargs=3, metavar=('TRAIN', 'VALID', 'TEST'))
    p.add_argument('--utterances-per-speaker', type=int)
    p.add_argument('--speech-archive', help='Reuse a locally downloaded train-clean-5.tar.gz; published MD5 is checked')
    return p


def main():
    p = parser(); args = p.parse_args()
    try:
        prepare(args)
    except KeyboardInterrupt:
        p.exit(130, 'Data preparation interrupted; no new catalog published.\n')
    except (PreparationError, OSError, tarfile.TarError, KeyError) as exc:
        p.exit(1, f'Data preparation stopped: {exc}\n')


if __name__ == '__main__':
    main()
