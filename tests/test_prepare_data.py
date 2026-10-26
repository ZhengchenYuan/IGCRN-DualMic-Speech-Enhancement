"""Offline fixtures exercise the pipeline; generated audio is not public-corpus evidence."""
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import prepare_data as prep
from scripts.simulate import simulate
from igcrn.data import Mixtures, rows

# Publisher license statements, abbreviated only for parser fixtures.
SLR_LICENSE = '<div>License: CC BY 4.0</div>'
ESC_LICENSE = ('The ESC-10 subset is licensed as a Creative Commons Attribution 3.0 Unported\n'
               '(https://creativecommons.org/licenses/by/3.0/) dataset.\n')

@pytest.fixture
def sources(tmp_path):
    archive = tmp_path/'speech.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for speaker in range(10, 16):
            for utterance in range(2):
                stream = io.BytesIO()
                sf.write(stream, np.random.default_rng(speaker*10+utterance).normal(0,.1,16000),
                         16000, format='FLAC')
                data = stream.getvalue()
                name = f'LibriSpeech/train-clean-5/{speaker}/20/{speaker}-20-{utterance:04d}.flac'
                member = tarfile.TarInfo(name); member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    metadata = 'filename,fold,target,category,esc10,src_file,take\n'
    license_text = ESC_LICENSE
    clips = {}
    for i in range(10):
        recording = str(1000+i); name = f'1-{recording}-A-10.wav'
        metadata += f'{name},1,10,rain,True,{recording},A\n'
        # These URLs and authors are deliberately fixture identifiers, never published provenance.
        license_text += (f'- [1-{recording}-A.ogg]:  clip derived from fixture.wav '
                         f'(https://example.invalid/sounds/{recording}/) by fixture-author [CC0]\n')
        audio = io.BytesIO()
        sf.write(audio,np.random.default_rng(i).normal(0,.1,44100),44100,format='WAV',subtype='PCM_16')
        clips[name] = audio.getvalue()
    return archive, metadata, license_text, clips


@pytest.mark.parametrize('mode,expected_count', [('smoke',10),('experiment',14)])
def test_offline_preparation_reproducible_and_simulator_compatible(tmp_path, sources, monkeypatch, mode, expected_count):
    archive, metadata, license_text, clips = sources
    fixture_md5 = prep.digest(archive, 'md5')
    monkeypatch.setattr(prep, 'SPEECH_MD5', fixture_md5)
    def fetch(url, path, expected=None, algorithm='sha256'):
        if url == prep.SLR_PAGE: data = SLR_LICENSE.encode()
        elif url.endswith('md5sum.txt'): data = f'{fixture_md5}  {prep.SPEECH_ARCHIVE}\n'.encode()
        elif url.endswith('/LICENSE'): data = license_text.encode()
        elif url.endswith('/esc50.csv'): data = metadata.encode()
        elif '/audio/' in url: data = clips[url.rsplit('/',1)[-1]]
        else: raise AssertionError(url)
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        return dict(url=url,sha256=prep.digest(path),bytes=len(data))
    monkeypatch.setattr(prep, 'download', fetch)
    out = tmp_path/'prepared'
    arguments=['--output',str(out),'--speech-archive',str(archive),'--mode',mode]
    if mode=='experiment': arguments += ['--speakers','2','1','1','--utterances-per-speaker','2','--noise-recordings','4','1','1']
    args=prep.parser().parse_args(arguments)
    prep.prepare(args); original=(out/'catalog.jsonl').read_bytes(); prep.prepare(args)
    assert (out/'catalog.jsonl').read_bytes()==original
    catalog=rows(out/'catalog.jsonl'); assert len(catalog)==expected_count
    for kind,key in [('speech','speaker_id'),('noise','recording_id')]:
        by_split=[{r[key] for r in catalog if r['kind']==kind and r['split']==split} for split in prep.SPLITS]
        assert all(by_split) and not (by_split[0]&by_split[1] or by_split[0]&by_split[2] or by_split[1]&by_split[2])
    for r in catalog:
        assert r['sha256']==prep.digest(r['path']) and r['selection_seed']==2026
        assert r['license'] in {'CC-BY-3.0','CC-BY-4.0'} and r['original_source_id']
        if r['kind']=='noise': assert r['original_license_label']=='CC0' and r['attribution_author']=='fixture-author'
    record=json.loads((out/'preparation.json').read_text())
    assert record['catalog_sha256']==hashlib.sha256(original).hexdigest()
    simulate(SimpleNamespace(catalog=str(out/'catalog.jsonl'),output=str(tmp_path/'mixtures'),seed=2026,count=3,seconds=.5))
    for split in prep.SPLITS:
        x,y=Mixtures(tmp_path/f'mixtures/{split}.jsonl')[0]
        assert x.shape==(2,8000) and y.shape==(8000,)
    args.seed=42
    with pytest.raises(prep.PreparationError,match='different selection'): prep.prepare(args)


def test_noise_original_identity_and_non_environment_filter(sources):
    _,meta,license_text,_=sources
    # A second take of the same original recording remains in one group.
    meta += '1-1000-B-10.wav,1,10,rain,True,1000,B\n'
    license_text += '- [1-1000-B.ogg]: clip derived from fixture.wav (https://example.invalid/1000) by fixture [CC-BY]\n'
    meta += '1-9999-A-10.wav,1,10,rain,False,9999,A\n'
    meta += '1-9998-A-0.wav,1,0,dog,True,9998,A\n'
    candidates=prep.noise_candidates(meta,license_text)
    assert len(candidates['1000'])==2 and len(candidates)==10
    assert set(prep.assign_groups(candidates,(2,1,1),2026,'noise-recordings'))<=set(candidates)


@pytest.mark.parametrize('failure',['speech','esc','checksum'])
def test_unverified_licenses_stop(failure):
    slr=SLR_LICENSE; esc=ESC_LICENSE; md5=f'{prep.SPEECH_MD5}  {prep.SPEECH_ARCHIVE}'
    if failure=='speech': slr='License unknown'
    elif failure=='esc': esc='Only the full ESC-50 noncommercial license is available'
    else: md5='0000 train-clean-5.tar.gz'
    with pytest.raises(prep.PreparationError): prep.verify_licenses(slr,esc,md5)


@pytest.mark.parametrize('change',['missing','unsupported'])
def test_unknown_clip_license_stops(sources,change):
    _,meta,license_text,_=sources
    license_text=ESC_LICENSE if change=='missing' else license_text.replace('[CC0]','[UNKNOWN]')
    with pytest.raises(prep.PreparationError,match='Manually confirm'):
        prep.noise_candidates(meta,license_text)


def test_deterministic_selection_and_insufficient_sources():
    groups={str(i):[] for i in range(100)}
    a=prep.assign_groups(groups,(6,2,2),2026,'test')
    assert a==prep.assign_groups(dict(reversed(list(groups.items()))),(6,2,2),2026,'test')
    assert a!=prep.assign_groups(groups,(6,2,2),2027,'test')
    with pytest.raises(prep.PreparationError,match='disjoint'): prep.assign_groups({},(1,1,1),0,'test')


@pytest.mark.parametrize('name,link',[('../escape.flac',False),('/absolute.flac',False),('symlink',True)])
def test_unsafe_archive_rejected(tmp_path,name,link):
    path=tmp_path/'bad.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        member=tarfile.TarInfo(name)
        if link: member.type=tarfile.SYMTYPE; member.linkname='../escape'
        tar.addfile(member)
    with tarfile.open(path) as tar, pytest.raises(prep.PreparationError,match='Unsafe'): prep.speech_index(tar)


def test_download_cache_and_corruption(tmp_path,monkeypatch):
    calls=[]
    def open_url(*args,**kwargs): calls.append(1); return io.BytesIO(b'original source')
    monkeypatch.setattr(prep.urllib.request,'urlopen',open_url)
    path=tmp_path/'download'; expected=hashlib.sha256(b'original source').hexdigest()
    prep.download('https://example.invalid/file',path,expected)
    prep.download('https://example.invalid/file',path,expected)
    assert len(calls)==1
    path.write_bytes(b'changed')
    with pytest.raises(prep.PreparationError,match='Checksum'): prep.download('https://example.invalid/file',path,expected)


def test_bad_download_never_published(tmp_path,monkeypatch):
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'bad bytes'))
    path=tmp_path/'download'
    with pytest.raises(prep.PreparationError,match='Publisher checksum'): prep.download('https://example.invalid/f',path,'abc')
    assert not path.exists() and not path.with_name(path.name+'.part').exists()


@pytest.mark.parametrize('arguments',[
    ['--utterances-per-speaker','0'], ['--speakers','1','0','1'], ['--seed','-1'],
    ['--output','/content/drive/MyDrive/IGCRN_NIO/raw'], ['--output','/content/drive'],
])
def test_invalid_args_stop_before_network(tmp_path,arguments):
    args=prep.parser().parse_args(['--output',str(tmp_path/'out'),*arguments])
    with pytest.raises(prep.PreparationError): prep.prepare(args)


def test_unknown_license_stops_before_audio_download(tmp_path, monkeypatch):
    calls=[]
    def fetch(url,path,*args):
        calls.append(url)
        values={prep.SLR_PAGE: 'License unknown', prep.SLR_BASE+'md5sum.txt': f'{prep.SPEECH_MD5}  {prep.SPEECH_ARCHIVE}',
                prep.ESC_BASE+'LICENSE': ESC_LICENSE, prep.ESC_BASE+'meta/esc50.csv': 'metadata fixture'}
        assert url in values, 'Audio must not be downloaded before license checks'
        Path(path).write_text(values[url]); return {'url':url}
    monkeypatch.setattr(prep,'download',fetch)
    out=tmp_path/'stopped'
    with pytest.raises(prep.PreparationError,match='Manually confirm'):
        prep.prepare(prep.parser().parse_args(['--output',str(out)]))
    assert len(calls)==4 and not (out/'catalog.jsonl').exists() and not (out/'raw').exists()
