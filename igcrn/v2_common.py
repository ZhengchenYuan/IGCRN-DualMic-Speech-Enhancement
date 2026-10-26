"""V2-only integrity helpers. Never rewrite legacy experiments."""
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def new_output(path, protected=()):
    path = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if path == source or source in path.parents:
            raise ValueError(f'Protected V1/input path: {path}')
    if path.exists():
        raise FileExistsError(f'Use a new output path: {path}')
    return path


def rng_for(seed, split, index, attempt=0, stream=0):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([seed, ('train','valid','test').index(split), index, attempt, stream]))


def start_budget(path, minutes):
    """Absolute deadline survives process/reconnect; never reset on resume."""
    path = Path(path)
    if path.exists():
        state = json.loads(path.read_text())
        if state['budget_minutes'] != minutes:
            raise ValueError('Cannot silently change an existing runtime budget')
        return state
    now = time.time()
    state = dict(schema_version=2, budget_minutes=minutes, started_unix=now, deadline_unix=now+minutes*60)
    json_write(path, state)
    return state
