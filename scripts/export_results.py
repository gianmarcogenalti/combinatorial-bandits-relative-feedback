#!/usr/bin/env python3
"""Validate every per-run regret curve and export reproducible reference results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import t


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_value(spec, key):
    if key == 'algorithm.name':
        return spec['algorithm_name']
    value = spec
    for part in key.split('.'):
        value = value[part]
    return value


def export(root, destination, repetitions, expected_runs, horizon=1000000):
    metadata = json.loads((root / 'metadata.json').read_text())
    active = {}
    for variant in metadata['run_ids']:
        saved = json.loads((root / 'runs' / variant / 'metadata.json').read_text())
        scientific_id = saved['spec']['run_id']
        assert scientific_id not in active, 'Duplicate scientific run identity'
        active[scientific_id] = (variant, saved['spec'])
    assert len(active) == expected_runs, (len(active), expected_runs)
    records = {}
    for manifest in (root / 'analysis/cache/metrics').glob('*/cache.json'):
        cache = json.loads(manifest.read_text())
        spec = cache['identity']['spec']
        run_id = spec['run_id']
        if run_id not in active:
            continue
        assert run_id not in records, f'Multiple metric records for {run_id}'
        assert spec == active[run_id][1], 'Metric specification differs from selected run'
        arrays_path = manifest.with_name('result.npz')
        assert digest(arrays_path) == cache['files']['result.npz']
        with np.load(arrays_path, allow_pickle=False) as arrays:
            x, values = arrays['x'].copy(), arrays['values'].copy()
        assert x[-1] == spec['budget_steps'] == horizon
        assert np.isfinite(values).all() and np.all(np.diff(values) >= -1e-8)
        assert np.all(values >= -1e-8) and np.all(values <= x + 1e-6)
        progress = json.loads((root / 'runs' / active[run_id][0] / 'progress.json').read_text())
        assert progress['status'] == 'completed' and progress['step'] == horizon
        records[run_id] = (spec, x, values)
    assert set(records) == set(active), 'Some requested runs have no durable metric'
    analysis = json.loads((root / 'analysis/metadata.json').read_text())['metrics']['regret']
    destination.mkdir(parents=True, exist_ok=True)
    final = []
    verified_points = 0
    with np.load(root / 'analysis/regret.npz', allow_pickle=False) as aggregate:
        for entry in analysis:
            labels = entry['labels']
            selected = [(s, x, y) for s, x, y in records.values()
                        if all(label_value(s, key) == val for key, val in labels.items())]
            assert len(selected) == entry['count'] == repetitions
            assert sorted(s['repetition'] for s, _, _ in selected) == list(range(repetitions))
            x = selected[0][1]
            assert all(np.array_equal(x, item[1]) for item in selected)
            values = np.stack([item[2] for item in selected])
            mean = values.mean(axis=0)
            sem = (values.std(axis=0, ddof=1) / np.sqrt(repetitions)
                   if repetitions > 1 else np.full_like(mean, np.nan))
            prefix = entry['prefix']
            np.testing.assert_array_equal(x, aggregate[prefix + '_x'])
            np.testing.assert_allclose(mean, aggregate[prefix + '_mean'], rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(sem, aggregate[prefix + '_uncertainty'], rtol=1e-12, atol=1e-8)
            assert entry['uncertainty_kind'] == 'standard_error'
            half = float(t.ppf(.975, repetitions - 1) * sem[-1])
            row = dict(labels, mean=float(mean[-1]), sem=float(sem[-1]), count=repetitions,
                       ci95_lower=float(mean[-1]) - half, ci95_upper=float(mean[-1]) + half)
            final.append(row)
            verified_points += len(x)
    shutil.copy2(root / 'analysis/regret.csv', destination / 'regret.csv')
    label_keys = sorted({key for row in final for key in row}
                        - {'mean', 'sem', 'count', 'ci95_lower', 'ci95_upper'})
    keys = label_keys + ['mean', 'sem', 'count', 'ci95_lower', 'ci95_upper']
    with (destination / 'final_regret.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(final)
    shutil.copytree(root / 'analysis/figures', destination / 'figures', dirs_exist_ok=True)
    shutil.copy2(root / 'compute/summary.json', destination / 'compute.json')
    shutil.copy2(root / 'compute/summary.md', destination / 'compute.md')
    copied = [destination / name for name in ('regret.csv', 'final_regret.csv', 'compute.json', 'compute.md')]
    copied += [destination / 'figures' / p.relative_to(root / 'analysis/figures')
               for p in (root / 'analysis/figures').rglob('*') if p.is_file()]
    for path in copied:
        if path.suffix == '.csv':
            path.write_text(path.read_text())  # Canonical LF; preserve numeric fields verbatim.
    config_path = root / 'config.resolved.yml'
    provenance = {
        'seed': 104729, 'horizon': horizon, 'repetitions_per_setting': repetitions,
        'independent_runs': expected_runs, 'verified_aggregate_points': verified_points,
        'ews_commit': '4ba0dd60e43e8dabae5b5f879cde01f6765418db',
        'validation': 'All requested per-run metric checksums, completion states, repetition identities, means and sample SEM recomputed and checked.',
        'confidence_level': .95, 'confidence_interval_formula': 'mean +/- t.ppf(.975, count-1) * SEM',
        'resolved_config_sha256': digest(config_path),
        'ews_provenance': metadata.get('provenance', {}),
        'csv_serialization': 'LF line endings; numeric fields preserved verbatim',
        'exporter_sha256': digest(Path(__file__)),
        'source_modules_sha256': {str(p): digest(p) for p in sorted(Path('experiment_code').glob('*.py'))},
        'files': {str(p.relative_to(destination)): digest(p) for p in sorted(copied)},
    }
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'runs': expected_runs, 'verified_points': verified_points, 'final': final}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=1000000)
    args = parser.parse_args()
    export(args.output, args.destination, args.repetitions, args.runs, args.horizon)
