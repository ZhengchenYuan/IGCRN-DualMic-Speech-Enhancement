"""Build a deterministic source catalog from downloaded public corpora."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--librispeech',required=True); p.add_argument('--musan',required=True)
p.add_argument('--output',required=True); a=p.parse_args(); result=[]
for folder,split in [('train-clean-100','train'),('dev-clean','valid'),('test-clean','test')]:
    files=sorted((Path(a.librispeech)/folder).rglob('*.flac'))
    if not files: raise ValueError(f'Missing {folder}')
    for f in files:
        result.append(dict(path=str(f.resolve()),kind='speech',split=split,speaker_id=f.stem.split('-')[0],
                           license='CC BY 4.0',source_url='https://www.openslr.org/12'))
noise=sorted((Path(a.musan)/'noise').rglob('*.wav'))
if len(noise)<10: raise ValueError('Need at least 10 MUSAN noise files')
for i,f in enumerate(noise):
    # Recordings, not chunks, are split. Frozen alphabetical ordering.
    split='train' if i%10<8 else ('valid' if i%10==8 else 'test')
    result.append(dict(path=str(f.resolve()),kind='noise',split=split,recording_id=f.stem,
                       license='CC BY 4.0; retain MUSAN attribution files',source_url='https://www.openslr.org/17'))
Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in result))
